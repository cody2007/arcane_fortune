use std::os::raw::{c_float};
use crate::layers::*;
use crate::model::*;
use super::f32_to_f16;

pub struct Weights {
	pub w_desc: cudnnTensorDescriptor_t,
	pub w_mem: gpuMem_t,
	
	pub dw_desc: cudnnTensorDescriptor_t,
	pub dw_mem: gpuMem_t,
	
	pub len: size_t,
	pub data_type: cudnnDataType_t
}

/////////////// fns to update weights
pub fn grad_descent(weights_vec: Vec<Weights>, model: &Model) {
	for weights in weights_vec.iter() {
		unsafe {cudnnAddTensor(model.handle.cudnn_val,
				model.eps(weights.data_type),
				weights.dw_desc, weights.dw_mem,
				
				model.one(weights.data_type),
				weights.w_desc, weights.w_mem)}.chk_err();
	}
}

pub fn rms_descent(weights_vec: Vec<Weights>, eps: f32, layer: &mut Layer) {
	for weights in weights_vec.iter() {
		const ALPHA: c_float = 0.9;
		const DENOM_EPS: c_float = 1e-10;
		
		macro_rules! rms_update{($weights_rms_tmp: expr) => {
			match weights.data_type {
				cudnnDataType_t::CUDNN_DATA_FLOAT => {
					unsafe {rms_update(ALPHA, eps, DENOM_EPS,
						weights.dw_mem, $weights_rms_tmp.val, weights.w_mem, weights.len)};
				} _ => {panic!("unsupported data type for optimizer");}
			}
		};};
		
		// weights_rms_tmp initialized
		if let Some(weights_rms_tmp) = &layer.weights_rms_tmp {
			rms_update!(weights_rms_tmp);
		}else{
			let weights_rms_tmp = gpuMem::new(layer.data_type, weights.len);
			weights_rms_tmp.zero_out();
			
			rms_update!(weights_rms_tmp);
			layer.weights_rms_tmp = Some(weights_rms_tmp);
		}
	}
}

pub fn adam_descent(weights_vec: Vec<Weights>, step: u64, eps: f32, layer: &mut Layer) {
	for weights in weights_vec.iter() {
		const BETA1: c_float = 0.9;
		const BETA2: c_float = 0.98;
		const DENOM_EPS: c_float = 1e-9;
		
		let t = step as f32;
		
		// αt = eps * sqrt( 1 − β2^t ) /(1 − β1^t ) 
		let a_t = if step != 0 {
			eps * (1. - (BETA2.powf(t))).sqrt() / (1. - (BETA1.powf(t)))
		}else {0.};
		
		//println!("{} {} {} {} {}", a_t, t, (1. - (BETA2.powf(t))).sqrt(), (1. - (BETA1.powf(t))),
		//		BETA2.powf(t));
		
		macro_rules! adam_update{($weights_adam: expr) => {
			match weights.data_type {
				cudnnDataType_t::CUDNN_DATA_FLOAT => {
					unsafe {adam_update(a_t, BETA1, BETA2, DENOM_EPS,
						weights.dw_mem, $weights_adam.m.val, $weights_adam.v.val,
						weights.w_mem, weights.len)};
				} cudnnDataType_t::CUDNN_DATA_HALF => {
					unsafe {adam_update_f16(a_t, BETA1, BETA2, DENOM_EPS,
						weights.dw_mem, $weights_adam.m.val, $weights_adam.v.val,
						weights.w_mem, weights.len)};
				} _ => {panic!("unsupported data type");}
			}
		};};
		
		// weights_adam initialized
		if let Some(weights_adam) = &layer.weights_adam {
			adam_update!(weights_adam);
		}else{
			let weights_adam = WeightsAdam {
				m: gpuMem::new(cudnnDataType_t::CUDNN_DATA_FLOAT, weights.len),
				v: gpuMem::new(cudnnDataType_t::CUDNN_DATA_FLOAT, weights.len)
			};
			
			weights_adam.m.zero_out();
			weights_adam.v.zero_out();
			
			adam_update!(weights_adam);
			layer.weights_adam = Some(weights_adam);
		}
	}
}

//////////////////////////////////////////////////////////////
/////////////// fns for manipulating weights & gradients

// return weight tensor lengths (in `w_tensor_lens`) from input weight_vecs
// ex. if a layer computes y = x*w + b,    w_tensor_lens could be [3*3, 3] for the weights & bias
pub fn ret_weight_lens(weights_vec: Vec<Weights>, w_tensor_lens: &mut Vec<usize>) {
	w_tensor_lens.clear();
	for weights in weights_vec.iter() {
		w_tensor_lens.push(weights.len);
	}
}

// related to the above function:
// w_tensor_ind indexes into w_tensor_lens[], val_ind indexes
// the weight tensor
//	computes: w[w_tensor_ind][val_ind] += eps
pub fn update_weight(mut weights_vec: Vec<Weights>, w_tensor_ind: usize, val_ind: usize, eps: f32) {
	let wv = &mut weights_vec[w_tensor_ind];
	assert!(wv.len > val_ind);
	let mut weights: Vec<f32> = ret_raw(&wv.w_mem, wv.len);
	weights[val_ind] += eps;
	set_raw(&mut wv.w_mem, &weights, wv.len);
}

// outputs dw into `dw`
pub fn ret_dw(weights_vec: Vec<Weights>, w_tensor_ind: usize, dw: &mut Vec<f32>) {
	let wv = &weights_vec[w_tensor_ind];
	let dw_ret: Vec<f32> = ret_raw(&wv.dw_mem, wv.len);
	dw.clear();
	for val in dw_ret.iter() {
		dw.push(*val);
	}
}

////////////////////////////////////////////////////////////////


/////////////////// run forward & backward then update weights
impl Model {
	pub fn grad_descent(&mut self, layer_ind: usize) {
		self.batch += 1;
		let req_layers = self.get_req_layers(layer_ind);
		
		self.reset_fwd_cache_flags();
		self.zero_out_gradients();
		self.forward_training(layer_ind);
		
		if let Output::Tensor(dy) = &self.layers[layer_ind].dy {
			match dy.mem.dataType {
				cudnnDataType_t::CUDNN_DATA_FLOAT => {dy.set(&vec!{1. as f32});}
				cudnnDataType_t::CUDNN_DATA_HALF => {dy.set(&vec!{f32_to_f16(1.)});}
				_ => {panic!("unsupported data type");}
			}
		}

		self.backward(&req_layers);
		
		// update weights
		for layer_ind in req_layers.iter() {
			let layer = &self.layers[*layer_ind];
			run_composite!{layer.internals => grad_descent(gradients(), self)};
		}
	}
	
	pub fn rms_descent(&mut self, layer_ind: usize) {
		self.batch += 1;
		let req_layers = self.get_req_layers(layer_ind);
		
		self.reset_fwd_cache_flags();
		self.zero_out_gradients();
		self.forward_training(layer_ind);
		
		if let Output::Tensor(dy) = &self.layers[layer_ind].dy {
			match dy.mem.dataType {
				cudnnDataType_t::CUDNN_DATA_FLOAT => {dy.set(&vec!{1. as f32});}
				cudnnDataType_t::CUDNN_DATA_HALF => {dy.set(&vec!{f32_to_f16(1.)});}
				_ => {panic!("unsupported data type");}
			}
		}
		
		self.backward(&req_layers);
		
		// update weights
		for layer_ind in req_layers.iter() {
			let layer = &mut self.layers[*layer_ind];
			run_composite!{layer.internals => rms_descent(gradients(),
					self.eps_f32[0], layer)};
		}
	}
	
	pub fn adam_descent(&mut self, layer_ind: usize) {
		self.batch += 1;
		let req_layers = self.get_req_layers(layer_ind);
		
		self.reset_fwd_cache_flags();
		self.zero_out_gradients();
		self.forward_training(layer_ind);
		
		if let Output::Tensor(dy) = &self.layers[layer_ind].dy {
			match dy.mem.dataType {
				cudnnDataType_t::CUDNN_DATA_FLOAT => {dy.set(&vec!{1. as f32});}
				cudnnDataType_t::CUDNN_DATA_HALF => {dy.set(&vec!{f32_to_f16(1.)});}
				_ => {panic!("unsupported data type");}
			}
		}
		
		self.backward(&req_layers);
		
		// update weights
		for layer_ind in req_layers.iter() {
			let layer = &mut self.layers[*layer_ind];
			run_composite!{layer.internals => adam_descent(gradients(),
					self.batch - 1, self.eps_f32[0], layer)};
		}	
	}
}

///////////////////// update weights only
impl Model {
	pub fn rms_update_weights_only(&mut self, req_layers: &Vec<usize>) {
		for layer_ind in req_layers.iter() {
			let layer = &mut self.layers[*layer_ind];
			run_composite!{layer.internals => rms_descent(gradients(),
					self.eps_f32[0], layer)};
		}
		self.batch += 1;
	}
	
	pub fn adam_update_weights_only(&mut self, req_layers: &Vec<usize>) {
		for layer_ind in req_layers.iter() {
			let layer = &mut self.layers[*layer_ind];
			run_composite!{layer.internals => adam_descent(gradients(),
					self.batch, self.eps_f32[0], layer)};
		}
		self.batch += 1;
	}
}

