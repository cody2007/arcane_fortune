#![allow(non_snake_case)]
use super::*;
use crate::layers::MulSoftmaxQKAndVParams;
use std::cmp::min;
use crate::saving::{KeyPair, find_req_key_parse};

/*
	Y[h, img, time1, vec_out] += 
		for time2 in 0..TIME2
			softmaxQK[h, img, time1, time2] * V[h, img, time2, vec_out]
		
		where h is the head
		
		Note: dimensions h & img can be thought of as a single dimension of size H*IMG
		
	Y[0,1,3] = softmaxQK[0,1,2] * V[0,2,3]
	
	==========================
		 dY[0,1,3]
		----------- = V[0,2,3]
		dSQV[0,1,2]
		
		
		dY[0,1,3]
		--------- = SQV[0,1,2]
		dV[0,2,3]
		 
			(derivatives are 0 when dimensions of numerator and denom
			 don't match)
			 
	=============================
		    dE            dE       dY[0,1,3]
		----------- = --------- * -----------
		dSQV[0,1,2]   dY[0,1,3]   dSQV[0,1,2]
		
				  dE
			  =  --------- * V[0,2,3]
			     dY[0,1,3]
		
		dEdQ[0,1,2] = dEdY[0,1,3] * V[0,2,3]
		
	=================================
		   dE          dE       dY[0,1,3]
		--------- = --------- * ---------
		dV[0,2,3]   dY[0,1,3]   dV[0,2,3]
		
				  dE
			  =  --------- * SQV[0,1,2]
			     dY[0,1,3]
		
		dEdK[0,2,3] = SQV[0,1,2] * dEdY[0,1,3]
*/

pub struct MulSoftmaxQKAndVInternals {params: MulSoftmaxQKAndVParams}

impl Run for MulSoftmaxQKAndVInternals {
	fn forward(&self, layer: &Layer, model: &Model) {
		#[cfg(feature="profile")]
		let _g = Guard::new("mul softmaxQK and V fwd");
		
		debug_assert!(layer.x_layers.len() == 2);
		
		let softmaxQK = &model.layers[layer.x_layers[0]].y.tensor();
		let V = &model.layers[layer.x_layers[1]].y.tensor();
		let y = &layer.y.tensor();
		
		let batch_sz = V.shape.n * V.shape.c; // H*IMG
		let N = softmaxQK.shape.h; // time1
		let K = V.shape.h; // time2
		let M = V.shape.w; // vec_out
		
		debug_assert!(N == K);
		debug_assert!((softmaxQK.shape.n * softmaxQK.shape.c) == batch_sz);
		debug_assert!((y.shape.n * y.shape.c) == batch_sz);
		
		model.einsum(y, &[0,1,3], softmaxQK, &[0,1,2], V, &[0,2,3], N, K, M, batch_sz, 0., self.params.data_type);
	}
	
	fn backward(&self, layer: &Layer, model: &Model) {
		#[cfg(feature="profile")]
		let _g = Guard::new("mul softmaxQK and V bwd");

		debug_assert!(layer.x_layers.len() == 2);
		
		let softmaxQK = &model.layers[layer.x_layers[0]].y.tensor();
		let V = &model.layers[layer.x_layers[1]].y.tensor();
		
		let dsoftmaxQK = &model.layers[layer.x_layers[0]].dy.tensor();
		let dV = &model.layers[layer.x_layers[1]].dy.tensor();
		
		let dy = &layer.dy.tensor();
		
		let batch_sz = V.shape.n * V.shape.c; // H*IMG
		
		debug_assert!((softmaxQK.shape.n * softmaxQK.shape.c) == batch_sz);
		debug_assert!((dy.shape.n * dy.shape.c) == batch_sz);

		/////////////////////////// dsoftmaxQK
		{
			let N = softmaxQK.shape.h; // time1
			let K = V.shape.w; // vec_out
			let M = V.shape.h; // time2
			
			debug_assert!(N == M);
			
			model.einsum(dsoftmaxQK, &[0,1,2], dy, &[0,1,3], V, &[0,2,3], N, K, M, batch_sz, 1., self.params.data_type);
		}
		
		///////////////////////// dV
		{
			let N = V.shape.h; // time2
			let K = softmaxQK.shape.h; // time1
			let M = V.shape.w; // vec_out
			
			debug_assert!(N == K);
			
			model.einsum(dV, &[0,2,3], softmaxQK, &[0,1,2], dy, &[0,1,3], N, K, M, batch_sz, 1., self.params.data_type);
		}
	}

	fn sv_arch(&self, txt: &mut String) {
		txt.push_str(&format!("\tdata_type: {}\n", self.params.data_type));
	}
}

impl Model {
	pub fn add_mul_softmaxQK_and_V(&mut self, V_layer_ind: usize,
			params: MulSoftmaxQKAndVParams) {
		let data_type = params.data_type;
		
		debug_assert!(self.layers.len() > 1);
		let softmax_layer_ind = self.layers.len() - 1;
		
		let softmaxQK_shape = self.layers[softmax_layer_ind].y.ravel_time_shape();
		let V_shape = self.layers[V_layer_ind].y.ravel_time_shape();
		
		// V[h, img, n_time, vec_out]
		let h = V_shape.n;
		let batch_sz = V_shape.c;
		let n_time = V_shape.h;
		let vec_out = V_shape.w;
		
		// softmaxQK[h, img, n_time, n_time]
		assert!(softmaxQK_shape.n == h);
		assert!(softmaxQK_shape.c == batch_sz);
		assert!(softmaxQK_shape.h == n_time);
		assert!(softmaxQK_shape.w == n_time);
		
		let y_shape = TensorShape {
			n: h,
			c: batch_sz,
			h: n_time,
			w: vec_out
		};
		
		let y = Tensor::new(data_type, y_shape);
		
		self.layers.push(
			Layer::new(vec![softmax_layer_ind, V_layer_ind],
				InternalTypes::MulSoftmaxQKAndV(MulSoftmaxQKAndVInternals {params}),
				y,
				String::from("MulSoftmaxQKAndV"),
				data_type
			));
	}
	
	pub fn load_mul_softmaxQK_and_V(&mut self, x_layers: &Vec<usize>, layer_keys: &Vec<KeyPair>) {
		assert!(x_layers.len() == 2);
		let V_layer_ind = min(x_layers[0], x_layers[1]);
		self.add_mul_softmaxQK_and_V(V_layer_ind, MulSoftmaxQKAndVParams {
			data_type: find_req_key_parse("data_type", layer_keys)
		});
	}
}
