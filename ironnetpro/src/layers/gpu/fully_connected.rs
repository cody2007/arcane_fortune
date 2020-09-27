#![allow(non_snake_case)]
use super::*;
use crate::layers::FullyConnectedParams;
use crate::saving::{KeyPair, find_req_key_parse};

/*
	Y[batch_sz, time, 1, vec_out, 1] += 
		for vec_in = 0..VEC_IN
			X[batch_sz, time, 1, vec_in, 1] * W[vec_out, 1, vec_in, 1] 
	
	Y[0,2] = X[0,1] * W[2,1]
		(batch_sz and time can be thought of as one raveled together dimension)
	
	================================
		 dY[0,2]
		--------- = W[2,1]
		 dX[0,1]
		
		
		dY[0,2]
		--------- = X[0,1]
		dW[2,1]
		 
			(derivatives are 0 when dimensions of numerator and denom
			 don't match)
			 
	=============================
		  dE          dE       dY[0,2]
		------- = --------- * ---------
		dX[0,1]     dY[0,2]    dX[0,1]
		
				  dE
			  =  --------- * W[2,1]
			      dY[0,2]
		
		dEdX[0,1] = dEdY[0,2] * W[2,1]
		
	=================================
		  dE          dE         dY[0,2]
		------- = ----------- * ---------
		dW[2,1]     dY[0,2]      dW[2,1]
		
				  dE
			  =  --------- * X[0,1]
			     dY[0,2]
		
		dEdW[2,1] = dEdY[0,2] * X[0,1]
*/

pub struct FullyConnectedInternals {
	params: FullyConnectedParams,
	
	pub W: Filter,
	pub dW: Filter
}

impl Run for FullyConnectedInternals {
	fn forward(&self, layer: &Layer, model: &Model) {
		#[cfg(feature="profile")]
		let _g = Guard::new("fc fwd");
		
		debug_assert!(layer.x_layers.len() == 1);
		
		let x = &model.layers[layer.x_layers[0]].y.tensor(); // output of input layer is the input for this layer
		let y = &layer.y.tensor();
		
		assert!(x.mem.dataType == y.mem.dataType && y.mem.dataType == self.W.mem.dataType);
		
		let batch_sz = 1;
		let N = x.shape.n * x.shape.c; // img*t
		let K = x.shape.h * x.shape.w; // vec_in
		let M = self.W.shape.k; // vec_out
		
		model.einsum(y, &[0,2], x, &[0,1], &self.W, &[2,1], N, K, M, batch_sz, 0.);
	}
	
	fn backward(&self, layer: &Layer, model: &Model) {
		#[cfg(feature="profile")]
		let _g = Guard::new("fc bwd");
		
		let in_layer = &model.layers[layer.x_layers[0]]; // output of input layer is the input for this layer
		
		let dx = &in_layer.dy.tensor(); 
		let x = &in_layer.y.tensor();
		let dy = &layer.dy.tensor();
		let batch_sz = 1;
		
		assert!(dx.mem.dataType == dy.mem.dataType && dy.mem.dataType == x.mem.dataType);

		//println!("x {} w {} y {}", x.shape.to_string(), self.W.shape.to_string(), dy.shape.to_string());
		
		/////////////////////////////////// dx
		{
			let N = x.shape.n * x.shape.c; // img*t
			let K = self.W.shape.k; // vec_out
			let M = x.shape.h * x.shape.w; // vec_in
			
			model.einsum(dx, &[0,1], dy, &[0,2], &self.W, &[2,1], N, K, M, batch_sz, 1.);
		}
		
		/////////////////////////////////// dW
		{
			let N = self.W.shape.k; // vec_out
			let K = x.shape.n * x.shape.c; // img*t
			let M = x.shape.h * x.shape.w; // vec_in
			
			model.einsum(&self.dW, &[2,1], dy,  &[0,2], x, &[0,1], N, K, M, batch_sz, 1.);
		}
	}
	
	fn zero_out_internal_gradients(&self) {
		self.dW.zero_out();
	}

	fn gradients(&self) -> Vec<Weights> {
		vec![Weights {
			w_desc: self.W.tensor_desc.val, 
			w_mem: self.W.mem.val,
			dw_desc: self.dW.tensor_desc.val,
			dw_mem: self.dW.mem.val,
			len: self.W.mem.n_elements,
			data_type: self.W.mem.dataType
		}]
	}
	
	fn sv_arch(&self, txt: &mut String) {
		txt.push_str(&format!("\tvec_out_sz: {}\n", self.params.vec_out_sz));
		txt.push_str(&format!("\tnorm_scale: {}\n", self.params.norm_scale));
		txt.push_str(&format!("\tdata_type: {}\n", self.params.data_type));
	}
	
	fn sv_weights(&self, _: &Layer, _: &Model, save_dir: &str, file_nm: &str) {
		self.W.sv(save_dir, file_nm);
	}
	
	fn sv_gradients(&self, save_dir: &str, file_nm: &str) {
		self.dW.sv(save_dir, file_nm);
	}
	
	fn ld_weights(&self, _: &Layer, _: &Model, save_dir: &str, file_nm: &str) {
		self.W.ld(save_dir, file_nm);
	}
}

impl Model {
	pub fn add_fully_connected(&mut self, params: FullyConnectedParams, rng: &mut XorState) {
		let data_type = params.data_type;
		
		let x_layer_ind = self.layers.len() - 1;
		
		let x = &self.layers[x_layer_ind].y;
		
		let x_shape = x.ravel_time_shape();
		// ^ [batch_sz, time, vec_in, 1] = [n,c,h,w]
		
		let vec_in = x_shape.h * x_shape.w;
		
		let y_shape = TensorShape {
			n: x_shape.n,
			c: x_shape.c,
			h: params.vec_out_sz,
			w: 1
		};
		
		let W_shape = FilterShape {
			k: params.vec_out_sz,
			c: 1,
			h: vec_in,
			w: 1
		};
		
		let y = Tensor::new(data_type, y_shape);
		
		let W = Filter::new_norm(data_type, W_shape, params.norm_scale, rng);
		let dW = Filter::zeros(data_type, W_shape);
		
		self.layers.push(
			Layer::new(vec![x_layer_ind],
				InternalTypes::FullyConnected(FullyConnectedInternals {
					params, W, dW
				}),
				y,
				String::from("FullyConnected"),
				data_type
			));
	}
	
	pub fn load_fully_connected(&mut self, layer_keys: &Vec<KeyPair>, rng: &mut XorState) {
		self.add_fully_connected(FullyConnectedParams {
			vec_out_sz: find_req_key_parse("vec_out_sz", layer_keys),
			norm_scale: find_req_key_parse("norm_scale", layer_keys),
			data_type: find_req_key_parse("data_type", layer_keys),
		}, rng);
	}
}

