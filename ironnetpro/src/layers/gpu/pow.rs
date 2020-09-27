use super::*;
use crate::saving::{KeyPair, find_req_key_parse};

#[allow(non_camel_case_types)]
type DATA_TYPE = f32;

pub struct PowInternals {params: PowParams}

impl Run for PowInternals {
	fn forward(&self, layer: &Layer, model: &Model) {
		#[cfg(feature="profile")]
		let _g = Guard::new("pow fwd");

		let x = model.layers[layer.x_layers[0]].y.tensor(); // output of input layer is the input for this layer
		let y = layer.y.tensor();
		
		unsafe {pow_forward(
				x.mem.val,
				self.params.alpha,
				y.mem.val,
				x.shape.n_elements() as size_t)};
	}
	
	fn backward(&self, layer: &Layer, model: &Model) {
		#[cfg(feature="profile")]
		let _g = Guard::new("pow bwd");

		let layer_input = &model.layers[layer.x_layers[0]];
		
		let x = layer_input.y.tensor();
		let dx = layer_input.dy.tensor();
		let dy = layer.dy.tensor();
		
		unsafe {pow_backward(
				x.mem.val,
				self.params.alpha,
				dy.mem.val,
				dx.mem.val,
				x.shape.n_elements() as size_t)};
	}
	
	fn sv_arch(&self, txt: &mut String) {
		txt.push_str(&format!("\talpha: {}\n", self.params.alpha));
		txt.push_str(&format!("\tdata_type: {}\n", self.params.data_type));
	}
}

pub struct PowParams {
	pub alpha: DATA_TYPE,
	pub data_type: cudnnDataType_t
}

impl Model {
	pub fn add_pow_layer_ind(&mut self, params: PowParams, layer_prev_ind: usize) {
		let data_type = params.data_type;
		debug_assert!(self.layers.len() > 0);
		
		let x = self.layers[layer_prev_ind].y.tensor(); // (input to this layer is the output of the previous layer)
		
		debug_assert!(x.mem.dataType == data_type);
		debug_assert!(data_type.bytes() == size_of::<DATA_TYPE>(), "alpha, beta for add op should be changed or generalized");
		
		let y = Tensor::new(data_type, x.shape);
		
		self.layers.push( Layer::new(
			vec![layer_prev_ind],
			InternalTypes::Pow(PowInternals {params}),
			y,
			String::from("pow"),
			data_type
		));
	}
	
	pub fn add_pow(&mut self, params: PowParams) {
		debug_assert!(self.layers.len() > 0);
		
		let layer_prev_ind = self.layers.len() - 1;
		self.add_pow_layer_ind(params, layer_prev_ind);
	}
	
	pub fn load_pow(&mut self, x_layers: &Vec<usize>, layer_keys: &Vec<KeyPair>) {
		let layer_prev_ind = x_layers[0];
		self.add_pow_layer_ind(PowParams {
				alpha: find_req_key_parse("alpha", layer_keys),
				data_type: find_req_key_parse("data_type", layer_keys)
			}, layer_prev_ind);
	}
}

