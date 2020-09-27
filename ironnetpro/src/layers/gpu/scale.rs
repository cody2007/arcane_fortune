use super::*;
use crate::layers::ScaleParams;
use std::ffi::c_void;
use crate::saving::{KeyPair, find_req_key_parse};

pub struct ScaleInternals {params: ScaleParams}

impl Run for ScaleInternals {
	fn forward(&self, layer: &Layer, model: &Model) {
		#[cfg(feature="profile")]
		let _g = Guard::new("scale fwd");

		debug_assert!(layer.x_layers.len() == 1);
		
		let x = &model.layers[layer.x_layers[0]].y.tensor();
		let y = &layer.y.tensor();
		
		//////////////// copy x to y
		debug_assert!(self.params.data_type == cudnnDataType_t::CUDNN_DATA_FLOAT &&
				  x.mem.dataType == cudnnDataType_t::CUDNN_DATA_FLOAT); // doesn't strictly require float, but that x & y be the same
		
		debug_assert!(y.mem.bytes == x.mem.bytes);
		
		unsafe {cudaMemcpy(y.mem.val as *mut c_void,
					 x.mem.val as *const c_void,
					 x.mem.bytes,
					 cudaMemcpyKind::cudaMemcpyDeviceToDevice)}.chk_err();
		
		/////////////// scale
		unsafe {cudnnScaleTensor(model.handle.cudnn_val,
				y.desc.val, y.mem.val,
				vec![self.params.alpha].as_ptr() as *const c_void)}.chk_err();
	}
	
	fn backward(&self, layer: &Layer, model: &Model) {
		#[cfg(feature="profile")]
		let _g = Guard::new("scale bwd");

		debug_assert!(layer.x_layers.len() == 1);
		
		let dx = &model.layers[layer.x_layers[0]].dy.tensor();
		let dy = &layer.dy.tensor();
		
		//////////////// copy add dy to dx
		debug_assert!(self.params.data_type == cudnnDataType_t::CUDNN_DATA_FLOAT &&
				  dx.mem.dataType == cudnnDataType_t::CUDNN_DATA_FLOAT); // doesn't strictly require float, but that x & y be the same
		
		debug_assert!(dy.mem.bytes == dx.mem.bytes);
		
		unsafe {cudnnAddTensor(model.handle.cudnn_val,
					vec![self.params.alpha].as_ptr() as *const c_void,
					dy.desc.val, dy.mem.val,
					
					model.one(layer.data_type),
					dx.desc.val, dx.mem.val)}.chk_err();
	}
	
	fn sv_arch(&self, txt: &mut String) {
		txt.push_str(&format!("\talpha: {}\n", self.params.alpha));
		txt.push_str(&format!("\tdata_type: {}\n", self.params.data_type));
	}
}

impl Model {
	pub fn add_scale(&mut self, params: ScaleParams) {
		let data_type = params.data_type;
		debug_assert!(self.layers.len() > 0);
		
		let layer_prev = self.layers.len() - 1;
		let x_shape = self.layers[layer_prev].y.ravel_time_shape();
		
		self.layers.push( Layer::new(
			vec![layer_prev],
			InternalTypes::Scale(ScaleInternals {params}),
			Tensor::new(data_type, x_shape),
			String::from("Scale"),
			data_type
		));
	}
	
	pub fn load_scale(&mut self, layer_keys: &Vec<KeyPair>) {
		self.add_scale(ScaleParams {
			alpha: find_req_key_parse("alpha", layer_keys),
			data_type: find_req_key_parse("data_type", layer_keys)
		});
	}
}
