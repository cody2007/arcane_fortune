use super::*;
use crate::saving::{KeyPair, find_req_key_parse};
use std::cmp::min;

pub struct MulInternalsCPU {}

impl RunCPU for MulInternalsCPU {
	fn forward(&mut self, layer_ind: usize, layers: &mut Vec<LayerCPU>) {
		let layer = &layers[layer_ind];
		debug_assert!(layer.x_layers.len() == 2);
		
		let x1 = &layers[layer.x_layers[0]].y; // output of input layer is the input for this layer
		let x1_mem = x1.mem();
		
		let x2 = &layers[layer.x_layers[1]].y; // output of input layer is the input for this layer
		let x2_mem = x2.mem();
		
		let x1_len = x1.ravel_time_shape().n_elements();
		let x2_len = x2.ravel_time_shape().n_elements();
		debug_assert!(x1_len == x2_len);
			
		let mut y = Vec::with_capacity(x1_len);
		for (x1v, x2v) in x1_mem.iter().take(x1_len).zip(x2_mem.iter()) {
			y.push(x1v * x2v);
		}
		
		*layers[layer_ind].y.mem_mut() = y;
	}
}

impl ModelCPU {
	pub fn add_mul(&mut self, layer2_ind: usize) {
		debug_assert!(self.layers.len() > 0);
		
		let layer_prev_ind = self.layers.len() - 1;
		
		debug_assert!(layer_prev_ind != layer2_ind);
		let x1 = &self.layers[layer_prev_ind].y; // (input to this layer is the output of the previous layer)
		let x2 = &self.layers[layer2_ind].y;
		
		let x1_shape = x1.ravel_time_shape();
		let x2_shape = x2.ravel_time_shape();
		
		let y_shape = TensorShape::broadcast(x1_shape, x2_shape);
		let y = TensorCPU::new(y_shape);
		
		// cudnnOpTensor expects the second tensor to be the smaller, if broadcasting
		// is to be performed (ie. x1: (3,4,5,6), x2: (3,1,1,6) is valid but
		// reversing the input parameters (x1 as x2 and x2 as x1) results in an error)
		if x1_shape.n > x2_shape.n || x1_shape.c > x2_shape.c ||
				   x1_shape.h > x2_shape.h || x1_shape.w > x2_shape.w {
			debug_assert!(x1_shape.n >= x2_shape.n && x1_shape.c >= x2_shape.c &&
					  x1_shape.h >= x2_shape.h && x1_shape.w >= x2_shape.w);
			self.new_layer(
				vec![layer_prev_ind, layer2_ind],
				InternalTypesCPU::Mul(MulInternalsCPU {}),
				y,
				String::from("mul"),
			);
		}else{
			debug_assert!(x2_shape.n >= x1_shape.n && x2_shape.c >= x1_shape.c &&
					  x2_shape.h >= x1_shape.h && x2_shape.w >= x1_shape.w);
			 self.new_layer(
				vec![layer2_ind, layer_prev_ind],
				InternalTypesCPU::Mul(MulInternalsCPU {}),
				y,
				String::from("mul")
			);
		}
	}
	
	pub fn load_mul(&mut self, x_layers: &Vec<usize>, layer_keys: &Vec<KeyPair>) {
		let layer2_ind = min(x_layers[0], x_layers[1]);
		
		let data_type = find_req_key_parse::<cudnnDataType_t>("data_type", &layer_keys);
		debug_assert!(data_type == cudnnDataType_t::CUDNN_DATA_FLOAT);
		
		self.add_mul(layer2_ind);
	}
}

