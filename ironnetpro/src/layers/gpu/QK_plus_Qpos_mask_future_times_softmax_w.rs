#![allow(non_snake_case)]
use super::*;
use crate::layers::QKPlusQPosMaskFutureTimesSoftmaxWParams;
use crate::saving::{KeyPair, find_req_key_parse};
use std::os::raw::c_void;

//	1. QK 	(output shape: y[h,img,time1,time2], inputs both: x[h,img,time1,vec_out)
//	2. Q*pos 	(output shape: ", pos: x[h,vec_out, time2]
// 	3. Q*K + Q*pos 	(both of shape: x]h,img,time1,time2]
//	4. mask future time: for all time2 > time1, set to negative infinity
//	5. scale outputs
//	6. softmax across w dimension (time2)

/*	Step 1:
 	Y[h, img, time1, time2] +=
		for vec_out in 0..VEC_OUT
			Q[h, img, time1, vec_out] * K[h, img, time2, vec_out]
			
		where h is the head
		
		Note: dimensions h & img can be thought of as
			a single dimension of size H*IMG
			
	Y[0,1,3] = Q[0,1,2] * K[0,3,2]
	
	================================
		dY[0,1,3]
		--------- = K[0,3,2]
		dQ[0,1,2]
		
		
		dY[0,1,3]
		--------- = Q[0,1,2]
		dK[0,3,2]
		 
			(derivatives are 0 when dimensions of numerator and denom
			 don't match)
			 
	=============================
		  dE            dE      dY[0,1,3]
		--------- = --------- * ---------
		dQ[0,1,2]   dY[0,1,3]   dQ[0,1,2]
		
				  dE
			  =  --------- * K[0,3,2]
			     dY[0,1,3]
		
		dEdQ[0,1,2] = dEdY[0,1,3] * K[0,3,2]
		
	=================================
		   dE          dE       dY[0,1,3]
		--------- = --------- * ---------
		dK[0,3,2]   dY[0,1,3]   dK[0,3,2]
		
				  dE
			  =  --------- * Q[0,1,2]
			     dY[0,1,3]
		
		dEdK[0,3,2] = dEdY[0,1,3] * Q[0,1,2]
*/

/* 	Step 2:
	Multiplies Q layer with learned relative position encodings
	
	First computes Q*R which produces:
	
	Y[h, img, :,:] = | P P P 0 |
	                 | P P 1 0 |
	                 | P 2 1 0 |
	                 | 3 2 1 0 |   matrix is: [time1, time2]
	     
	     where the numbers represent the rel distance (from R) that was multiplied with Q
	     and P represents garbage points (future time points) that should be removed.
	     We shift each row (for each img and head) to remove all `P`s, because only values of
	     t2 <= t1 (i.e., the lower left triangle) are used after the mask_future_times layer.
	     
	The shifting occurs in place and results in:
	
	Y[h, img, :,:] = | 0 U U U |
	                 | 1 0 U U |
	                 | 2 1 0 U |
	                 | 3 2 1 0 |   matrix is: [time1, time2]
	      
	      where U are undefined values (currently just the previous values)
	
	------------------
	
	Y[h, img, time1, time2] +=
	  for vec_out = 0..VEC_OUT
		Q[h, img, time1, vec_out] * R[h, vec_out, time2]
		
		where h is the head
		
		Note: dimensions img & time1 can be thought of as 
			a single dimension of size IMG*T
	
	Y[0,1,3] = Q[0,1,2] * R[0,2,3]
	
	dEdQ[0,1,2] = dEdY[0,1,3] * R[0,2,3]
	
	dEdR[0,2,3] = dEdY[0,1,3] * Q[0,1,2]
	
	See layers/gpu/mul_Q_K.rs and other multi-headed attn
	files for more verbose examples of this notation.
*/

const SOFTMAX_ALG: cudnnSoftmaxAlgorithm_t = cudnnSoftmaxAlgorithm_t::CUDNN_SOFTMAX_ACCURATE;
const MODE: cudnnSoftmaxMode_t = cudnnSoftmaxMode_t::CUDNN_SOFTMAX_MODE_INSTANCE; // compute across all values for each img

pub struct QKPlusQPosMaskFutureTimesSoftmaxWInternals {
	params: QKPlusQPosMaskFutureTimesSoftmaxWParams,
	softmax_desc: TensorDescriptor,
		
	pub R: Filter, // aka the position, `pos`
	pub dR: Filter
}

impl Run for QKPlusQPosMaskFutureTimesSoftmaxWInternals {
	fn forward(&self, layer: &Layer, model: &Model) {
		#[cfg(feature="profile")]
		let _g = Guard::new("mask future times add fwd");
		
		debug_assert!(layer.x_layers.len() == 2);
		
		let Q = &model.layers[layer.x_layers[0]].y.tensor();
		let y = &layer.y.tensor();
		
		let n_heads = Q.shape.n;
		let n_imgs = Q.shape.c;
		let n_time = Q.shape.h;
		let vec_out = Q.shape.w;

		debug_assert!(Q.shape.h == n_time);
		debug_assert!(self.params.data_type == cudnnDataType_t::CUDNN_DATA_FLOAT &&
				  Q.mem.dataType == cudnnDataType_t::CUDNN_DATA_FLOAT);
		
		// Q*pos (needs to be performed first because results must be shifted)
		{
			let batch_sz = n_heads;
			let N = n_imgs * n_time; // time1
			let K = vec_out;
			let M = n_time; // time2
			
			model.einsum(y, &[0,1,3], Q, &[0,1,2], &self.R, &[0,2,3], N, K, M, batch_sz, 0., self.params.data_type);
			unsafe {shift_QR_pos_lleft_triangle(y.mem.val, (n_heads*n_imgs) as usize, n_time as usize)};
		}
		
		// Q*K (added to Q*pos result)
		{
			let key = &model.layers[layer.x_layers[1]].y.tensor();
			
			let batch_sz = Q.shape.n * Q.shape.c; // H*IMG
			let N = n_time; // time1
			let K = vec_out;
			let M = n_time; // time2
			
			debug_assert!(Q.shape == key.shape);
			debug_assert!(key.mem.dataType == key.mem.dataType);
			
			model.einsum(y, &[0,1,3], Q, &[0,1,2], key, &[0,3,2], N, K, M, batch_sz, 1., self.params.data_type);
		}
		
		// mask future times & scale by params.scale
		let y_tmp = model.shared_workspace.as_ref().unwrap();
		{
			let n_exemplars = Q.shape.n * Q.shape.c;
			
			unsafe {raw::mask_future_times(y_tmp.val, y.mem.val, self.params.scale,
					n_exemplars as size_t, n_time as size_t)};
		}
		
		// softmax
		unsafe {raw::cudnnSoftmaxForward(
					model.handle.cudnn_val, SOFTMAX_ALG, MODE,
					
					model.one(layer.data_type),
					self.softmax_desc.val, y_tmp.val,
					
					model.zero(layer.data_type),
					self.softmax_desc.val, y.mem.val)}.chk_err();
	}
	
	fn backward(&self, layer: &Layer, model: &Model) {
		#[cfg(feature="profile")]
		let _g = Guard::new("mask future times add bwd");
		
		debug_assert!(layer.x_layers.len() == 2);
		
		let dx1 = &model.layers[layer.x_layers[0]].dy.tensor();
		let dx2 = &model.layers[layer.x_layers[1]].dy.tensor();
		let dy = &layer.dy.tensor();
		let y = &layer.y.tensor();
		
		debug_assert!(self.params.data_type == cudnnDataType_t::CUDNN_DATA_FLOAT &&
				  dx1.mem.dataType == cudnnDataType_t::CUDNN_DATA_FLOAT &&
				  dx2.mem.dataType == dx1.mem.dataType);
		// the following code doesn't strictly require that the data are floats, only that dx & dy are the same datatype
		
		// in place softmax backward (updates dy)
		unsafe {raw::cudnnSoftmaxBackward(
					model.handle.cudnn_val, SOFTMAX_ALG, MODE,
					
					vec![self.params.scale].as_ptr() as *const c_void,
					self.softmax_desc.val, y.mem(),
					self.softmax_desc.val, dy.mem(),
					
					model.zero(layer.data_type),
					self.softmax_desc.val, dy.mem())}.chk_err();
		
		// mask future times has no backward step because the softmax results in zero outputs
		
		// QK*K + Q*pos   ==> dy identically propagated back to Q*K and Q*pos
		
		///////////////// Q*K
		{
			debug_assert!(layer.x_layers.len() == 2);
			
			let query = &model.layers[layer.x_layers[0]].y.tensor();
			let key = &model.layers[layer.x_layers[1]].y.tensor();
			
			let dquery = &model.layers[layer.x_layers[0]].dy.tensor();
			let dkey = &model.layers[layer.x_layers[1]].dy.tensor();
			
			let dy = &layer.dy.tensor();
			
			debug_assert!(query.shape == key.shape);
			debug_assert!(dquery.shape == dkey.shape && query.shape == dquery.shape);
			
			let batch_sz = query.shape.n * query.shape.c; // H*IMG
			
			///////////////////////// dQ
			{
				let N = query.shape.h; // time1
				let K = query.shape.h; // time2
				let M = query.shape.w; // vec_out
				
				model.einsum(dquery, &[0,1,2], dy, &[0,1,3], key, &[0,3,2], N, K, M, batch_sz, 1., self.params.data_type);
			}
			
			///////////////////////// dK
			{
				let N = key.shape.h; // time2
				let K = key.shape.h; // time1
				let M = key.shape.w; // vec_out
				
				model.einsum(dkey, &[0,3,2], dy, &[0,1,3], query, &[0,1,2], N, K, M, batch_sz, 1., self.params.data_type);
			}
		}
		
		///////////////// Q*pos (NOTE: alters dy)
		{
			let in_layer = &model.layers[layer.x_layers[0]]; // output of input layer is the input for this layer
			
			let dQ = &in_layer.dy.tensor(); 
			let Q = &in_layer.y.tensor();
			let dy = &layer.dy.tensor();
			
			// Q[h, img, time1, vec_out]
			// R[h, vec_out, time2]
			
			let n_heads = Q.shape.n;
			let n_imgs = Q.shape.c;
			let n_time = Q.shape.h;
			let vec_out = Q.shape.w;
			
			let batch_sz = n_heads;
			debug_assert!(n_heads == self.R.shape.k); // h
			debug_assert!(n_time == self.R.shape.h); // time1 == time2
			debug_assert!(vec_out == self.R.shape.c); // vec_out == vec_out
			
			unsafe {shift_QR_pos_uright_triangle(dy.mem.val, (n_heads*n_imgs) as usize, n_time as usize)};
			
			/////////////////////////////////// dQ
			{
				let N = n_imgs * n_time; // time1
				let K = n_time; // time2
				let M = vec_out;
				
				model.einsum(dQ, &[0,1,2], dy, &[0,1,3], &self.R, &[0,2,3], N, K, M, batch_sz, 1., self.params.data_type);
			}
			
			// sv dQ
			//dQ.ret().sv("/tmp/", "dQ");
			//println!("dQ {}", dQ.shape.to_string());
			
			/////////////////////////////////// dR
			{
				let N = vec_out;
				let K = n_imgs * n_time; // time1
				let M = n_time; // time2
				
				model.einsum(&self.dR, &[0,2,3], Q,  &[0,1,2], dy, &[0,1,3], N, K, M, batch_sz, 1., self.params.data_type);
			}
		}
	}
	
	fn zero_out_internal_gradients(&self) {
		self.dR.zero_out();
	}
	
	fn gradients(&self) -> Vec<Weights> {
		vec![Weights {
			w_desc: self.R.tensor_desc.val, 
			w_mem: self.R.mem.val,
			dw_desc: self.dR.tensor_desc.val,
			dw_mem: self.dR.mem.val,
			len: self.R.mem.n_elements,
			data_type: self.R.mem.dataType
		}]
	}
	
	fn sv_arch(&self, txt: &mut String) {
		txt.push_str(&format!("\tweight_initialization: {}\n", self.params.weight_initialization));
		txt.push_str(&format!("\tscale: {}\n", self.params.scale));
		txt.push_str(&format!("\tdata_type: {}\n", self.params.data_type));
	}
	
	fn sv_weights(&self, _: &Layer, _: &Model, save_dir: &str, file_nm: &str) {
		self.R.sv(save_dir, file_nm);
	}
	
	fn sv_gradients(&self, save_dir: &str, file_nm: &str) {
		self.dR.sv(save_dir, file_nm);
	}
	
	fn ld_weights(&self, _: &Layer, _: &Model, save_dir: &str, file_nm: &str) {
		self.R.ld(save_dir, file_nm);
	}
}

impl Model {
	pub fn add_QK_plus_Qpos_mask_future_times_softmax_w(&mut self, Q_layer_ind: usize, K_layer_ind: usize,
			params: QKPlusQPosMaskFutureTimesSoftmaxWParams, rng: &mut XorState) {
		let data_type = params.data_type;
		debug_assert!(self.layers.len() > 0);
		
		let Q_shape = self.layers[Q_layer_ind].y.ravel_time_shape();
		let K_shape = self.layers[K_layer_ind].y.ravel_time_shape();
		
		assert!(Q_shape == K_shape);
		
		let n_heads = Q_shape.n;
		let batch_sz = Q_shape.c;
		let n_time = Q_shape.h;
		let vec_out = Q_shape.w;
		
		assert!(n_heads != 1 && batch_sz != 1 && n_time != 1 && vec_out != 1);
		
		// position
		let (R, dR) = {
			let R_shape = FilterShape {
				k: n_heads,
				c: vec_out,
				h: n_time,
				w: 1
			};
			
			let R = Filter::new_init(data_type, R_shape, params.weight_initialization, rng);
			//let R = Filter::zeros(data_type, R_shape);
			let dR = Filter::zeros(data_type, R_shape);
			
			(R, dR)
		};
		
		// softmax descriptors (also allocates workspace for fwd pass)
		let softmax_desc = {
			let softmax_shape = TensorShape {
				n: n_heads*batch_sz*n_time,
				c: 1,
				h: 1,
				w: n_time
			};
				
			// used w/ softmax forward because it cannot work in-place w/ same input & output buffer
			self.allocate_shared_workspace(params.data_type, softmax_shape.n_elements());
			
			TensorDescriptor::new(params.data_type, TENSOR_FORMAT, softmax_shape)
		};
		
		let y_shape = TensorShape {
			n: n_heads,
			c: batch_sz,
			h: n_time,
			w: n_time
		};

		self.layers.push( Layer::new(
			vec![Q_layer_ind, K_layer_ind],
			InternalTypes::QKPlusQPosMaskFutureTimesSoftmaxW(
				QKPlusQPosMaskFutureTimesSoftmaxWInternals {
						params,
						softmax_desc,
						R, dR
				}),
			Tensor::new(data_type, y_shape),
			String::from("QKPlusQPosMaskFutureTimesSoftmaxW"),
			data_type
		));
	}
	
	pub fn load_QK_plus_Qpos_mask_future_times_softmax_w(&mut self, x_layers: &Vec<usize>, layer_keys: &Vec<KeyPair>, rng: &mut XorState) {
		assert!(x_layers.len() == 2);
		
		self.add_QK_plus_Qpos_mask_future_times_softmax_w(x_layers[0], x_layers[1], QKPlusQPosMaskFutureTimesSoftmaxWParams {
			weight_initialization: find_req_key_parse("weight_initialization", layer_keys),
			scale: find_req_key_parse("scale", layer_keys),
			data_type: find_req_key_parse("data_type", layer_keys)
		}, rng);
	}
}

