use super::*;
use crate::saving::{KeyPair, find_req_key_parse};
use crate::layers::MaxPoolParams;

pub struct PoolingInternals {
	params: MaxPoolParams,
	pooling_desc: PoolingDescriptor
}

const MAX_POOLING_MODE: cudnnPoolingMode_t = cudnnPoolingMode_t::CUDNN_POOLING_MAX;

impl Run for PoolingInternals {
	fn forward(&self, layer: &Layer, model: &Model) {
		let x = model.layers[layer.x_layers[0]].y.tensor(); // output of input layer is the input for this layer
		let y = layer.y.tensor();
		
		unsafe {cudnnPoolingForward(model.handle.cudnn_val,
				self.pooling_desc.val,
				
				model.one(layer.data_type),
				x.desc.val, x.mem.val,
				
				model.zero(layer.data_type),
				y.desc.val, y.mem.val)}.chk_err();
	}
	
	fn backward(&self, layer: &Layer, model: &Model) {
		let layer_prev = &model.layers[layer.x_layers[0]];
		
		let x = layer_prev.y.tensor();
		let dx = layer_prev.dy.tensor();
		
		let y = layer.y.tensor();
		let dy = layer.dy.tensor();
		
		unsafe {cudnnPoolingBackward(model.handle.cudnn_val,
				self.pooling_desc.val,
				
				model.one(layer.data_type),
				y.desc.val, y.mem.val,
				dy.desc.val, dy.mem.val,
				x.desc.val, x.mem.val,
				
				model.one(layer.data_type),
				dx.desc.val, dx.mem.val)}.chk_err();
	}
	
	fn sv_arch(&self, txt: &mut String) {
		txt.push_str(&format!("\tpool_sz: {}\n", self.params.pool_sz));
		txt.push_str(&format!("\tpad_h: {}\n", self.params.pad_h));
		txt.push_str(&format!("\tpad_w: {}\n", self.params.pad_w));
		txt.push_str(&format!("\tstride: {}\n", self.params.stride));
		txt.push_str(&format!("\tdata_type: {}\n", self.params.data_type));
	}
}

impl Model {
	pub fn add_max_pooling(&mut self, params: MaxPoolParams) {
		let data_type = params.data_type;
		debug_assert!(self.layers.len() > 0);
		
		let layer_prev_ind = self.layers.len() - 1;
		let layer_prev = &self.layers[layer_prev_ind];
		
		let x = layer_prev.y.tensor(); // (input to this layer is the output of the previous layer)
		
		let n_imgs = x.shape.n;
		let n_input_channels = x.shape.c;
		
		let pooling_desc = PoolingDescriptor::new(MAX_POOLING_MODE, NAN_PROP,
				params.pool_sz, params.pool_sz,
				params.pad_h, params.pad_w,
				params.stride, params.stride);
		
		// see documentation of cudnnGetPooling2dForwardOutputDim() (3.121, pg. 214) 
		let h = 1 + (x.shape.h + 2*params.pad_h - params.pool_sz)/params.stride;
		let w = 1 + (x.shape.w + 2*params.pad_w - params.pool_sz)/params.stride;
		
		let y_shape = TensorShape {n: n_imgs, c: n_input_channels, h, w};
		
		self.layers.push( Layer::new(
			vec!{layer_prev_ind; 1},
			InternalTypes::Pooling(PoolingInternals {
					params,
					pooling_desc
			}),
			Tensor::new(data_type, y_shape),
			String::from("max_pool"),
			data_type
		));
	}
	
	pub fn load_max_pooling(&mut self, layer_keys: &Vec<KeyPair>) {
		self.add_max_pooling(MaxPoolParams {
			pool_sz: find_req_key_parse("pool_sz", layer_keys),
			pad_h: find_req_key_parse("pad_h", layer_keys),
			pad_w: find_req_key_parse("pad_w", layer_keys),
			stride: find_req_key_parse("stride", layer_keys),
			data_type: find_req_key_parse("data_type", layer_keys)
		});
	}
}

