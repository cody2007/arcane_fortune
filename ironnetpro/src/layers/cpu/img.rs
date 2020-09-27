// this is a terminal (input) layer
use super::*;
use crate::layers::ImgParams;
use crate::saving::{KeyPair, find_req_key_parse};

pub struct ImgInternalsCPU {}

impl RunCPU for ImgInternalsCPU {
	fn forward(&mut self, _layer_ind: usize, _: &mut Vec<LayerCPU>) {}
}

impl ModelCPU {
	pub fn add_imgs(&mut self, params: ImgParams) {
		let shape = params.shape;
		
		self.new_layer(
				Vec::new(), // no inputs
				InternalTypesCPU::Img(ImgInternalsCPU {}),
				TensorCPU::new(shape),
				String::from("imgs")
		);
	}
	
	pub fn load_imgs(&mut self, layer_keys: &Vec<KeyPair>) {
		let data_type = find_req_key_parse("data_type", &layer_keys);
		debug_assert!(data_type == cudnnDataType_t::CUDNN_DATA_FLOAT);
		
		let mut shape: TensorShape = find_req_key_parse("shape", &layer_keys);
		shape.n = self.batch_sz;
		
		self.add_imgs(ImgParams {
			shape,
			data_type
		});
	}
}

