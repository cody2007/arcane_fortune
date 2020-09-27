use super::*;
use crate::saving::{KeyPair, find_req_key_parse};
use crate::cudnn_common::{cudnnSoftmaxAlgorithm_t};

pub struct SoftmaxInternalsCPU {
	alg: cudnnSoftmaxAlgorithm_t
}

impl RunCPU for SoftmaxInternalsCPU {
	fn forward(&mut self, layer_ind: usize, layers: &mut Vec<LayerCPU>) {
		let layer = &layers[layer_ind];
		debug_assert!(layer.x_layers.len() == 1);
		
		let x = &layers[layer.x_layers[0]].y; // output of input layer is the input for this layer
		let x_mem = x.mem();
		
		// shape of inputs/outputs
		let (n_imgs, vec_sz) = {
			debug_assert!(x.ravel_time_shape() == layer.y.ravel_time_shape());
			let shape = x.ravel_time_shape();
			(shape.n as usize, (shape.c * shape.h * shape.w) as usize)
		};
		
		// compute e^x
		let mut e_x = Vec::with_capacity(n_imgs*vec_sz);
		for xv in x_mem.iter().take(n_imgs*vec_sz) {
			e_x.push((*xv).exp());
		}
		
		let y_mem = &mut layers[layer_ind].y.mem_mut();
		
		// compute outputs
		for img in 0..n_imgs {
			let e_x_sum: f32 = e_x.iter().skip(img*vec_sz).take(vec_sz).sum();
			for (yv, e_xv) in y_mem.iter_mut().skip(img*vec_sz).take(vec_sz).zip(
				    e_x.iter().skip(img*vec_sz).take(vec_sz)){
				*yv = *e_xv / e_x_sum;
			}
		}
		
		// take natural log, if relevant
		match self.alg {
			cudnnSoftmaxAlgorithm_t::CUDNN_SOFTMAX_LOG => {
				for yv in y_mem.iter_mut().take(n_imgs*vec_sz) {
					*yv = (*yv).ln();
				}
			}
			cudnnSoftmaxAlgorithm_t::CUDNN_SOFTMAX_FAST |
			cudnnSoftmaxAlgorithm_t::CUDNN_SOFTMAX_ACCURATE => {}
		}
	}
}

impl ModelCPU {
	fn add_softmax_alg(&mut self, alg: cudnnSoftmaxAlgorithm_t, nm: String) {
		debug_assert!(self.layers.len() > 0);
		
		let layer_prev = self.layers.len() - 1;
		
		let x_shape = self.layers[layer_prev].y.ravel_time_shape(); // (input to this layer is the output of the previous layer)
		
		self.new_layer(
			vec!{layer_prev; 1},
			InternalTypesCPU::Softmax(SoftmaxInternalsCPU {alg}),
			TensorCPU::new(x_shape),
			nm
		);
	}
	
	pub fn add_softmax_log(&mut self) {
		self.add_softmax_alg(cudnnSoftmaxAlgorithm_t::CUDNN_SOFTMAX_LOG, String::from("softmax_log"));
	}
	
	pub fn load_softmax_log(&mut self, layer_keys: &Vec<KeyPair>) {
		let data_type = find_req_key_parse::<cudnnDataType_t>("data_type", &layer_keys);
		debug_assert!(data_type == cudnnDataType_t::CUDNN_DATA_FLOAT);
		
		self.add_softmax_log();
	}
}

