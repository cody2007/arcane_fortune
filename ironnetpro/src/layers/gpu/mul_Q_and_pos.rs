#![allow(non_snake_case)]
use super::*;
use crate::layers::MulQAndPosParams;
use crate::saving::{KeyPair, find_req_key_parse};

/*
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

pub struct MulQAndPosInternals {
	params: MulQAndPosParams,
	
	pub R: Filter,
	pub dR: Filter
}

impl Run for MulQAndPosInternals {
	fn forward(&self, layer: &Layer, model: &Model) {
		#[cfg(feature="profile")]
		let _g = Guard::new("mul Q and pos fwd");

		debug_assert!(layer.x_layers.len() == 1);
		
		let Q = &model.layers[layer.x_layers[0]].y.tensor(); // output of input layer is the input for this layer
		let y = &layer.y.tensor();
		
		// Q[h, img, time1, vec_out]
		// R[h, vec_out, time2]
		
		let n_heads = Q.shape.n;
		let n_imgs = Q.shape.c;
		let n_time = Q.shape.h;
		let vec_out = Q.shape.w;
		
		let batch_sz = n_heads;
		let N = n_imgs * n_time; // time1
		let K = vec_out;
		let M = n_time; // time2
		debug_assert!(n_heads == self.R.shape.k); // h
		debug_assert!(n_time == self.R.shape.h); // time1 == time2
		debug_assert!(vec_out == self.R.shape.c); // vec_out == vec_out
		
		model.einsum(y, &[0,1,3], Q, &[0,1,2], &self.R, &[0,2,3], N, K, M, batch_sz, 0.);
		
		unsafe {shift_QR_pos_lleft_triangle(y.mem.val, (n_heads*n_imgs) as usize, n_time as usize)};
	}
	
	fn backward(&self, layer: &Layer, model: &Model) {
		#[cfg(feature="profile")]
		let _g = Guard::new("mul Q and Pos bwd");
		
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
		
		//dy.ret().sv("/tmp/", "dy_pre");
		
		unsafe {shift_QR_pos_uright_triangle(dy.mem.val, (n_heads*n_imgs) as usize, n_time as usize)};
		
		//dy.ret().sv("/tmp/", "dy_post");

		/////////////////////////////////// dQ
		{
			let N = n_imgs * n_time; // time1
			let K = n_time; // time2
			let M = vec_out;
			
			model.einsum(dQ, &[0,1,2], dy, &[0,1,3], &self.R, &[0,2,3], N, K, M, batch_sz, 1.);
		}
		
		/////////////////////////////////// dR
		{
			let N = vec_out;
			let K = n_imgs * n_time; // time1
			let M = n_time; // time2
			
			model.einsum(&self.dR, &[0,2,3], Q,  &[0,1,2], dy, &[0,1,3], N, K, M, batch_sz, 1.);
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
		txt.push_str(&format!("\tnorm_scale: {}\n", self.params.norm_scale));
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
	pub fn add_mul_Q_and_pos(&mut self, Q_layer_ind: usize, params: MulQAndPosParams, rng: &mut XorState) {
		let data_type = params.data_type;
		
		let Q_layer = &self.layers[Q_layer_ind];
		
		let Q = &Q_layer.y;
		
		let Q_shape = Q.ravel_time_shape();
		// ^ [n_heads, batch_sz, time, vec_out] = [n,c,h,w]
		
		let n_heads = Q_shape.n;
		let n_imgs = Q_shape.c;
		let n_time = Q_shape.h;
		let vec_out = Q_shape.w;
		
		let y_shape = TensorShape {
			n: n_heads,
			c: n_imgs,
			h: n_time,
			w: n_time
		};
		
		let R_shape = FilterShape {
			k: n_heads,
			c: vec_out,
			h: n_time,
			w: 1
		};
		
		let y = Tensor::new(data_type, y_shape);
		
		let R = Filter::new_norm(data_type, R_shape, params.norm_scale, rng);
		let dR = Filter::zeros(data_type, R_shape);
		
		self.layers.push(
			Layer::new(vec![Q_layer_ind],
				InternalTypes::MulQAndPos(MulQAndPosInternals {
					params, R, dR
				}),
				y,
				String::from("MulQAndPos"),
				data_type
			));
	}
	
	pub fn load_mul_Q_and_pos(&mut self, x_layers: &Vec<usize>, layer_keys: &Vec<KeyPair>, rng: &mut XorState) {
		debug_assert!(x_layers.len() == 1);
		self.add_mul_Q_and_pos(x_layers[0], MulQAndPosParams {
			norm_scale: find_req_key_parse("norm_scale", layer_keys),
			data_type: find_req_key_parse("data_type", layer_keys)
		}, rng);
	}
}

