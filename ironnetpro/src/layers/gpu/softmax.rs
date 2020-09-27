use super::*;
use crate::layers::SoftmaxParams;
use crate::saving::{KeyPair, find_req_key_parse};

pub struct SoftmaxInternals {
	params: SoftmaxParams,
	alg: cudnnSoftmaxAlgorithm_t,
	opt_tensor_desc: Option<TensorDescriptor>
}

const MODE: cudnnSoftmaxMode_t = cudnnSoftmaxMode_t::CUDNN_SOFTMAX_MODE_INSTANCE; // compute across all values for each img

impl Run for SoftmaxInternals {	
	fn forward(&self, layer: &Layer, model: &Model) {
		#[cfg(feature="profile")]
		let _g = Guard::new("softmax fwd");

		debug_assert!(layer.x_layers.len() == 1);
		
		let x = &model.layers[layer.x_layers[0]].y; // output of input layer is the input for this layer
		let y = &layer.y;
		
		// custom shape (ex. to control / limit dimension over which softmax is taken)
		if let Some(tensor_desc) = &self.opt_tensor_desc {
			unsafe {raw::cudnnSoftmaxForward(
					model.handle.cudnn_val, self.alg, MODE,
					
					model.one(layer.data_type),
					tensor_desc.val, x.mem(),
					
					model.zero(layer.data_type),
					tensor_desc.val, y.mem())}.chk_err();
		}else{
			unsafe {raw::cudnnSoftmaxForward(
					model.handle.cudnn_val, self.alg, MODE,
					
					model.one(layer.data_type),
					x.ravel_time_tensor_desc(), x.mem(),
					
					model.zero(layer.data_type),
					y.ravel_time_tensor_desc(), y.mem())}.chk_err();
		}
	}
	
	fn backward(&self, layer: &Layer, model: &Model) {
		#[cfg(feature="profile")]
		let _g = Guard::new("softmax bwd");

		let dx = &model.layers[layer.x_layers[0]].dy; // output of input layer is the input for this layer
		let y = &layer.y;
		let dy = &layer.dy;
		
		// custom shape (ex. to control / limit dimension over which softmax is taken)
		if let Some(tensor_desc) = &self.opt_tensor_desc {
			unsafe {raw::cudnnSoftmaxBackward(
					model.handle.cudnn_val, self.alg, MODE,
					
					model.one(layer.data_type),
					tensor_desc.val, y.mem(),
					tensor_desc.val, dy.mem(),
					
					model.one(layer.data_type),
					tensor_desc.val, dx.mem())}.chk_err();
		}else{
			unsafe {raw::cudnnSoftmaxBackward(
					model.handle.cudnn_val, self.alg, MODE,
					
					model.one(layer.data_type),
					y.ravel_time_tensor_desc(), y.mem(),
					dy.ravel_time_tensor_desc(), dy.mem(),
					
					model.one(layer.data_type),
					dx.ravel_time_tensor_desc(), dx.mem())}.chk_err();
		}
	}
	
	fn sv_arch(&self, txt: &mut String) {
		txt.push_str(&format!("\tdata_type: {}\n", self.params.data_type));
	}
}

impl Model {
	fn add_softmax_alg(&mut self, params: SoftmaxParams, alg: cudnnSoftmaxAlgorithm_t, 
			opt_tensor_desc: Option<TensorDescriptor>, nm: String) {
		let data_type = params.data_type;
		debug_assert!(self.layers.len() > 0);
		
		let layer_prev = self.layers.len() - 1;
		
		let x_shape = self.layers[layer_prev].y.ravel_time_shape(); // (input to this layer is the output of the previous layer)
		
		self.layers.push( Layer::new(
			vec![layer_prev],
			InternalTypes::Softmax(SoftmaxInternals {params, alg, opt_tensor_desc}),
			Tensor::new(data_type, x_shape),
			nm,
			data_type
		));
	}
	
	///////////////////////////////////////////////////////
	// add utility functions
	
	// softmax across all values for each image
	pub fn add_softmax(&mut self, params: SoftmaxParams) {
		self.add_softmax_alg(params, cudnnSoftmaxAlgorithm_t::CUDNN_SOFTMAX_ACCURATE, None, String::from("softmax"));
	}
	
	// softmax across all values for each image
	pub fn add_softmax_log(&mut self, params: SoftmaxParams) {
		self.add_softmax_alg(params, cudnnSoftmaxAlgorithm_t::CUDNN_SOFTMAX_LOG, None, String::from("softmax_log"));
	}
	
	// softmax across only the w dimension
	pub fn add_softmax_across_w(&mut self, params: SoftmaxParams) {
		debug_assert!(TENSOR_FORMAT == cudnnTensorFormat_t::CUDNN_TENSOR_NCHW);
		
		let layer_prev_shape = self.layers.last().unwrap().y.ravel_time_shape();
		
		let tensor_desc = TensorDescriptor::new(params.data_type,
						TENSOR_FORMAT,
						TensorShape {
							n: layer_prev_shape.n * layer_prev_shape.c * layer_prev_shape.h,
							c: 1,
							h: 1,
							w: layer_prev_shape.w
						});
		
		self.add_softmax_alg(params, cudnnSoftmaxAlgorithm_t::CUDNN_SOFTMAX_ACCURATE, Some(tensor_desc), String::from("softmax_across_w"));
	}
	
	////////////////////////////////////////////////////
	// loading
	
	pub fn load_softmax(&mut self, layer_keys: &Vec<KeyPair>) {
		self.add_softmax(SoftmaxParams {
				data_type: find_req_key_parse("data_type", layer_keys)
		});
	}
	
	pub fn load_softmax_log(&mut self, layer_keys: &Vec<KeyPair>) {
		self.add_softmax_log(SoftmaxParams {
				data_type: find_req_key_parse("data_type", layer_keys)
		});
	}
	
	pub fn load_softmax_across_w(&mut self, layer_keys: &Vec<KeyPair>) {
		self.add_softmax_across_w(SoftmaxParams {
			data_type: find_req_key_parse("data_type", layer_keys)
		});
	}
}

