use std::os::raw::{c_double};
use super::*;
use crate::saving::{KeyPair, find_req_key_parse};
use crate::layers::ActivationParams;

pub struct ActivationInternals {
	params: ActivationParams,
	activation_desc: ActivationDescriptor
}

const RELU_COEF: c_double = 0.0;

impl Run for ActivationInternals {
	fn forward(&self, layer: &Layer, model: &Model) {
		#[cfg(feature="profile")]
		let _g = Guard::new("activation fwd");
		
		let x = model.layers[layer.x_layers[0]].y.tensor(); // output of input layer is the input for this layer
		let y = layer.y.tensor();
		
		unsafe {cudnnActivationForward(model.handle.cudnn_val,
				self.activation_desc.val,
				
				model.one(layer.data_type),
				x.desc.val, x.mem.val,
				
				model.zero(layer.data_type),
				y.desc.val, y.mem.val)}.chk_err();
	}
	
	fn backward(&self, layer: &Layer, model: &Model) {
		#[cfg(feature="profile")]
		let _g = Guard::new("activation fwd");
		
		let layer_prev = &model.layers[layer.x_layers[0]];
		
		let x = layer_prev.y.tensor();
		let dx = layer_prev.dy.tensor();
		let y = layer.y.tensor();
		let dy = layer.dy.tensor();
		
		unsafe {cudnnActivationBackward(model.handle.cudnn_val,
				self.activation_desc.val,
				
				model.one(layer.data_type),
				y.desc.val, y.mem.val,
				dy.desc.val, dy.mem.val,
				x.desc.val, x.mem.val,
				
				model.one(layer.data_type),
				dx.desc.val, dx.mem.val)}.chk_err();
	}
	
	fn sv_arch(&self, txt: &mut String) {
		txt.push_str(&format!("\tdata_type: {}\n", self.params.data_type));
	}
}

impl Model {
	pub fn add_relu(&mut self, params: ActivationParams) {
		self.add_activation(cudnnActivationMode_t::CUDNN_ACTIVATION_RELU, params);
	}
	
	pub fn add_tanh(&mut self, params: ActivationParams) {
		self.add_activation(cudnnActivationMode_t::CUDNN_ACTIVATION_TANH, params);
	}
	
	fn add_activation(&mut self, activation_mode: cudnnActivationMode_t, params: ActivationParams) {
		let data_type = params.data_type;
		debug_assert!(self.layers.len() > 0);
		
		let layer_prev_ind = self.layers.len() - 1;
		let layer_prev = &self.layers[layer_prev_ind];
		
		let x_shape = layer_prev.y.tensor().shape; // (input to this layer is the output of the previous layer)
		
		let activation_desc = ActivationDescriptor::new(activation_mode, NAN_PROP, RELU_COEF);
		
		let layer_nm = match activation_mode {
			cudnnActivationMode_t::CUDNN_ACTIVATION_RELU => "relu",
			cudnnActivationMode_t::CUDNN_ACTIVATION_TANH => "tanh",
			_ => {panic!("unknown activation mode");}
		};
		
		self.layers.push( Layer::new(
			vec!{layer_prev_ind; 1},
			InternalTypes::Activation(ActivationInternals {
					params,
					activation_desc
			}),
			Tensor::new(data_type, x_shape),
			String::from(layer_nm),
			data_type
		));
	}
	
	pub fn load_relu(&mut self, layer_keys: &Vec<KeyPair>) {
		self.add_relu(ActivationParams {
				data_type: find_req_key_parse("data_type", layer_keys)
		});
	}
	
	pub fn load_tanh(&mut self, layer_keys: &Vec<KeyPair>) {
		self.add_tanh(ActivationParams {
				data_type: find_req_key_parse("data_type", layer_keys)
		});
	}
}

