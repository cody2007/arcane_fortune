#![allow(non_snake_case)]
use super::*;
use crate::layers::QKVLayerParams;
use crate::saving::{KeyPair, find_req_key_parse};

/*
	QKV layers create Q, K, or V from X and weights for each head
	
	Y[h, img, t, vec_out] +=
	  for vec_in = 0..VEC_IN
		X[img, t, vec_in] * W[h, vec_in, vec_out]
		
		
		where Y is Q, K, or V,
			h is the head
			VEC_IN is the size of the vec_in dimension
		
		Note: dimensions img & t can be thought of as 
			a single dimension of size IMG*T
	
	Y[2,0,3] = X[0,1] * W[2,1,3]

	================================
		dY[2,0,3]
		--------- = W[2,1,3]
		 dX[0,1]
		
		
		dY[2,0,3]
		--------- = X[0,1]
		dW[2,1,3]
		 
			(derivatives are 0 when dimensions of numerator and denom
			 don't match)
			 
	=============================
		  dE          dE      dY[2,0,3]
		------- = --------- * ---------
		dX[0,1]   dY[2,0,3]    dX[0,1]
		
				  dE
			  =  --------- * W[2,1,3]
			     dY[2,0,3]
		
		dEdX[0,1] = dEdY[2,0,3] * W[2,1,3]
		
	=================================
		  dE          dE        dY[2,0,3]
		------- = ----------- * ---------
		dW[2,1,3]   dY[2,0,3]   dW[2,1,3]
		
				  dE
			  =  --------- * X[0,1]
			     dY[2,0,3]
		
		dEdW[2,1,3] = dEdY[2,0,3] * X[0,1]
*/

pub struct QKVInternals {
	params: QKVLayerParams,
	
	pub W: Filter,
	pub dW: Filter,
	
	op_tensor_desc: OpTensorDescriptor,
	
	pub bias: Filter,
	pub dbias: Filter,
	
	reduction_workspaces: ReductionWorkspaces
}

impl Run for QKVInternals {
	fn forward(&self, layer: &Layer, model: &Model) {
		#[cfg(feature="profile")]
		let _g = Guard::new("QKV fwd");
		
		debug_assert!(layer.x_layers.len() == 1);
		
		let x = &model.layers[layer.x_layers[0]].y.tensor(); // output of input layer is the input for this layer
		let y = &layer.y.tensor();
		
		let batch_sz = self.params.n_heads;
		let N = x.shape.n * x.shape.c; // img*t
		let K = x.shape.h; // vec_in
		let M = self.W.shape.h; // vec_out
		
		let y_tmp = model.shared_workspace.as_ref().unwrap();
		
		// y_tmp[2,0,3] = x[0,1] * W[2,1,3]
		model.einsum(&y_tmp.w_shape(y.shape()), &[2,0,3], x, &[0,1], &self.W, &[2,1,3], N, K, M, batch_sz, 0.);
		
		// y = y_tmp + bias (bias is broadcast added)
		unsafe {cudnnOpTensor(model.handle.cudnn_val,
				self.op_tensor_desc.val,
				
				model.one(layer.data_type),
				y.desc.val, y_tmp.val,
				
				model.one(layer.data_type),
				self.bias.tensor_desc.val, self.bias.mem.val,
				
				model.zero(layer.data_type),
				y.desc.val, y.mem.val)}.chk_err();
	}
	
	fn backward(&self, layer: &Layer, model: &Model) {
		#[cfg(feature="profile")]
		let _g = Guard::new("QKV bwd");
		
		let in_layer = &model.layers[layer.x_layers[0]]; // output of input layer is the input for this layer
		
		let dx = &in_layer.dy.tensor(); 
		let x = &in_layer.y.tensor();
		let dy = &layer.dy.tensor();
		
		/////// dbias
		{
			/*unsafe {cudnnReduceTensor(
					dy.mem.val,
					self.dbias.mem.val)}.chk_err();*/
			
			/*let n_batches = (dy.shape.n * dy.shape.c) as usize;
			let bias_sz = self.dbias.shape.n_elements();
			
			assert!((n_batches*bias_sz) == dy.shape.n_elements());
			
			unsafe {dbias_plus_dy(self.dbias.mem.val, dy.mem.val, n_batches, bias_sz)};*/
			
			unsafe {cudnnReduceTensor(model.handle.cudnn_val,
				self.reduction_workspaces.desc.val,
				self.reduction_workspaces.indices.val,
				self.reduction_workspaces.indices.bytes,
				self.reduction_workspaces.workspace.val,
				self.reduction_workspaces.workspace.bytes,
				
				model.one(layer.data_type),
				dy.desc.val, dy.mem.val,
				
				model.zero(layer.data_type),
				self.dbias.tensor_desc.val, self.dbias.mem.val)}.chk_err(); 
			// ^ fails w/ CUDNN_STATUS_INVALID_VALUE
		}

		/////////////////////////////////// dx
		{
			let batch_sz = 1;
			let N = x.shape.n * x.shape.c; // img*t
			let K = self.params.n_heads * self.W.shape.h; // h*vec_out
			let M = x.shape.h; // vec_in
			
			model.einsum(dx, &[0,1], dy, &[2,0,3], &self.W, &[2,1,3], N, K, M, batch_sz, 1.);
		}
		
		/////////////////////////////////// dW
		{
			let batch_sz = dy.shape.n;
			let N = x.shape.h; // vec_in
			let K = x.shape.n * x.shape.c; // img*t
			let M = self.W.shape.h; // vec_out
			
			model.einsum(&self.dW, &[2,1,3], x,  &[0,1], dy, &[2,0,3], N, K, M, batch_sz, 1.);
		}
	}
	
	fn zero_out_internal_gradients(&self) {
		self.dW.zero_out();
		self.dbias.zero_out();
	}
	
	fn gradients(&self) -> Vec<Weights> {
		vec![Weights {
			w_desc: self.W.tensor_desc.val, 
			w_mem: self.W.mem.val,
			dw_desc: self.dW.tensor_desc.val,
			dw_mem: self.dW.mem.val,
			len: self.W.mem.n_elements,
			data_type: self.W.mem.dataType
		    },
		    Weights {
		      w_desc: self.bias.tensor_desc.val, 
			w_mem: self.bias.mem.val,
			dw_desc: self.dbias.tensor_desc.val,
			dw_mem: self.dbias.mem.val,
			len: self.bias.mem.n_elements,
			data_type: self.bias.mem.dataType
		    }]
	}
	
	fn sv_arch(&self, txt: &mut String) {
		txt.push_str(&format!("\tweight_initialization: {}\n", self.params.weight_initialization));
		txt.push_str(&format!("\tbias_initialization: {}\n", self.params.bias_initialization));
		txt.push_str(&format!("\tn_heads: {}\n", self.params.n_heads));
		txt.push_str(&format!("\tdata_type: {}\n", self.params.data_type));
	}
	
	fn sv_weights(&self, _: &Layer, _: &Model, save_dir: &str, file_nm: &str) {
		self.W.sv(save_dir, &format!("{}_W", file_nm));
		self.bias.sv(save_dir, &format!("{}_bias", file_nm));
	}
	
	fn sv_gradients(&self, save_dir: &str, file_nm: &str) {
		self.dW.sv(save_dir, &format!("{}_W", file_nm));
		self.dbias.sv(save_dir, &format!("{}_bias", file_nm));
	}
	
	fn ld_weights(&self, _: &Layer, _: &Model, save_dir: &str, file_nm: &str) {
		self.W.ld(save_dir, &format!("{}_W", file_nm));
		self.bias.ld(save_dir, &format!("{}_bias", file_nm));
	}
	
	fn workspace_sz(&self) -> usize {
		self.reduction_workspaces.indices.bytes +
		self.reduction_workspaces.workspace.bytes
	}
}

impl Model {
	pub fn add_QKV_layer(&mut self, x_layer_ind: usize, params: QKVLayerParams, rng: &mut XorState) {
		let data_type = params.data_type;
		
		let x_layer = &self.layers[x_layer_ind];
		
		let x = &x_layer.y;
		
		let x_shape = x.ravel_time_shape();
		// ^ [img, time, vec_in, 1] = [n,c,h,w]
		
		let batch_sz = x_shape.n;
		let n_time = x_shape.c;
		let vec_in = x_shape.h;
		assert!(x_shape.w == 1);
		
		assert!((vec_in % params.n_heads) == 0, "vec_in {} should be divisible by n_heads {}", vec_in, params.n_heads);
		let vec_out = vec_in / params.n_heads;
		
		let y_shape = TensorShape {
			n: params.n_heads,
			c: batch_sz,
			h: n_time,
			w: vec_out
		};
		
		let W_shape = FilterShape {
			k: params.n_heads,
			c: vec_in,
			h: vec_out,
			w: 1
		};
		
		let bias_shape = FilterShape {
			k: params.n_heads,
			c: 1,
			h: 1,
			w: vec_out
		};
		
		let op_tensor_desc = OpTensorDescriptor::new(cudnnOpTensorOp_t::CUDNN_OP_TENSOR_ADD, data_type, NAN_PROP);
		
		let y = Tensor::new(data_type, y_shape);
		
		let W = Filter::new_init(data_type, W_shape, params.weight_initialization, rng);
		let dW = Filter::zeros(data_type, W_shape);
		
		//let bias = Filter::new_init(data_type, bias_shape, params.bias_initialization, rng);
		let bias = Filter::zeros(data_type, bias_shape);
		let dbias = Filter::zeros(data_type, bias_shape);
		
		// \/ used w/ forward because cudnnOpTensor cannot work in-place w/ same input & output buffer
		self.allocate_shared_workspace(params.data_type, y.shape.n_elements());
		
		let reduction_workspaces = ReductionWorkspaces::new(self, &y.desc, &bias.tensor_desc);
		
		self.layers.push(
			Layer::new(vec![x_layer_ind],
				InternalTypes::QKV(QKVInternals {
					params, op_tensor_desc,
					W, dW, bias, dbias,
					reduction_workspaces
				}),
				y,
				String::from("QKV"),
				data_type
			));
	}
	
	pub fn load_QKV_layer(&mut self, x_layers: &Vec<usize>, layer_keys: &Vec<KeyPair>, rng: &mut XorState) {
		assert!(x_layers.len() == 1);
		self.add_QKV_layer(x_layers[0], QKVLayerParams {
			weight_initialization: find_req_key_parse("weight_initialization", layer_keys),
			bias_initialization: find_req_key_parse("bias_initialization", layer_keys),
			n_heads: find_req_key_parse("n_heads", layer_keys),
			data_type: find_req_key_parse("data_type", layer_keys)
		}, rng);
	}
}

