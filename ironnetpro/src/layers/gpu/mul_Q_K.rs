#![allow(non_snake_case)]
use super::*;
use crate::layers::MulQKParams;
use crate::saving::{KeyPair, find_req_key_parse};

/*
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

pub struct MulQKInternals {params: MulQKParams}

impl Run for MulQKInternals {
	fn forward(&self, layer: &Layer, model: &Model) {
		#[cfg(feature="profile")]
		let _g = Guard::new("mul QK fwd");
		
		debug_assert!(layer.x_layers.len() == 2);
		
		let query = &model.layers[layer.x_layers[0]].y.tensor();
		let key = &model.layers[layer.x_layers[1]].y.tensor();
		let y = &layer.y.tensor();
		
		debug_assert!(query.shape == key.shape);
		
		let batch_sz = query.shape.n * query.shape.c; // H*IMG
		let N = query.shape.h; // time1
		let K = query.shape.w; // vec_out
		let M = query.shape.h; // time2
		
		model.einsum(y, &[0,1,3], query, &[0,1,2], key, &[0,3,2], N, K, M, batch_sz, 0.);
	}
	
	fn backward(&self, layer: &Layer, model: &Model) {
		#[cfg(feature="profile")]
		let _g = Guard::new("mul QK bwd");
		
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
			
			model.einsum(dquery, &[0,1,2], dy, &[0,1,3], key, &[0,3,2], N, K, M, batch_sz, 1.);
		}
		
		///////////////////////// dK
		{
			let N = key.shape.h; // time2
			let K = key.shape.h; // time1
			let M = key.shape.w; // vec_out
			
			model.einsum(dkey, &[0,3,2], dy, &[0,1,3], query, &[0,1,2], N, K, M, batch_sz, 1.);
		}
	}
	
	fn sv_arch(&self, txt: &mut String) {
		txt.push_str(&format!("\tdata_type: {}\n", self.params.data_type));
	}
}

impl Model {
	pub fn add_mul_Q_K_layer(&mut self, Q_layer_ind: usize, K_layer_ind: usize,
			params: MulQKParams) {
		let data_type = params.data_type;
		
		let Q_shape = self.layers[Q_layer_ind].y.ravel_time_shape();
		let K_shape = self.layers[K_layer_ind].y.ravel_time_shape();
		
		assert!(Q_shape == K_shape);
		
		let h = Q_shape.n;
		let batch_sz = Q_shape.c;
		let n_time = Q_shape.h;
		let vec_out = Q_shape.w;
		
		assert!(h != 1 && batch_sz != 1 && n_time != 1 && vec_out != 1);
		
		let y_shape = TensorShape {
			n: h,
			c: batch_sz,
			h: n_time,
			w: n_time
		};
		
		let y = Tensor::new(data_type, y_shape);
		
		self.layers.push(
			Layer::new(vec![Q_layer_ind, K_layer_ind],
				InternalTypes::MulQK(MulQKInternals {params}),
				y,
				String::from("MulQK"),
				data_type
			));
	}
	
	pub fn load_mul_Q_K_layer(&mut self, x_layers: &Vec<usize>, layer_keys: &Vec<KeyPair>) {
		assert!(x_layers.len() == 2);
		self.add_mul_Q_K_layer(x_layers[0], x_layers[1], MulQKParams {
			data_type: find_req_key_parse("data_type", layer_keys)
		});
	}
}

