// this is a terminal (input) layer
use super::*;
use crate::layers::ImgParams;
use crate::saving::{KeyPair, find_req_key_parse};

pub struct ImgInternals {params: ImgParams}

impl Run for ImgInternals {
	fn forward(&self, _: &Layer, _: &Model) {}
	fn backward(&self, _: &Layer, _: &Model) {}
	
	fn sv_arch(&self, txt: &mut String) {
		txt.push_str(&format!("\tshape: {}\n", self.params.shape.to_string()));
		txt.push_str(&format!("\tdata_type: {}\n", self.params.data_type));
	}
}

impl Model {
	pub fn add_imgs(&mut self, params: ImgParams) {
		let data_type = params.data_type;
		let shape = params.shape;
		
		self.layers.push( Layer::new(
						Vec::new(), // no inputs
						InternalTypes::Img(ImgInternals {params}),
						Tensor::new(data_type, shape),
						String::from("imgs"),
						data_type
		));
	}
	
	pub fn load_imgs(&mut self, layer_keys: &Vec<KeyPair>) {
		let shape: TensorShape = find_req_key_parse("shape", &layer_keys);
		//shape.n = self.batch_sz;
		
		self.add_imgs(ImgParams {
			data_type: find_req_key_parse("data_type", &layer_keys),
			shape
		});
	}
}

