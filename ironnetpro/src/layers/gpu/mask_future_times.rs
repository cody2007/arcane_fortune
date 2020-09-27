use super::*;
use crate::layers::MaskFutureTimesParams;
use crate::saving::{KeyPair, find_req_key_parse};
use std::os::raw::c_void;

pub struct MaskFutureTimesInternals {
	params: MaskFutureTimesParams
}

impl Run for MaskFutureTimesInternals {
	fn forward(&self, layer: &Layer, model: &Model) {
		#[cfg(feature="profile")]
		let _g = Guard::new("mask future times fwd");
		
		debug_assert!(layer.x_layers.len() == 1);
		
		let x = &model.layers[layer.x_layers[0]].y.tensor();
		let y = &layer.y.tensor();
		
		let n_time = x.shape.w;
		debug_assert!(x.shape.h == n_time);
		
		let n_exemplars = x.shape.n * x.shape.c;
		
		debug_assert!(self.params.data_type == cudnnDataType_t::CUDNN_DATA_FLOAT &&
				  x.mem.dataType == cudnnDataType_t::CUDNN_DATA_FLOAT);
		
		unsafe {raw::mask_future_times(y.mem.val, x.mem.val, self.params.scale,
				n_exemplars as size_t, n_time as size_t)};
	}
	
	fn backward(&self, layer: &Layer, model: &Model) {
		#[cfg(feature="profile")]
		let _g = Guard::new("mask future times bwd");
		
		debug_assert!(layer.x_layers.len() == 1);
		
		let dx = &model.layers[layer.x_layers[0]].dy.tensor();
		let dy = &layer.dy.tensor();
		
		debug_assert!(self.params.data_type == cudnnDataType_t::CUDNN_DATA_FLOAT &&
				  dx.mem.dataType == cudnnDataType_t::CUDNN_DATA_FLOAT);
		// the following code doesn't strictly require that the data are floats, only that dx & dy are the same datatype
		
		//dx.ret().sv("/tmp/", "dx_pre");
		
		unsafe {cudnnAddTensor(model.handle.cudnn_val,
					vec![self.params.scale].as_ptr() as *const c_void,
					dy.desc.val, dy.mem.val,
					
					model.one(layer.data_type),
					dx.desc.val, dx.mem.val)}.chk_err();
		
		/*println!("{}", dy.shape.to_string());
		println!("{}", dx.shape.to_string());
		
		dy.ret().sv("/tmp/", "dy");
		dx.ret().sv("/tmp/", "dx");
		panic!();*/
	}
	
	fn sv_arch(&self, txt: &mut String) {
		txt.push_str(&format!("\tscale: {}\n", self.params.scale));
		txt.push_str(&format!("\tdata_type: {}\n", self.params.data_type));
	}
}

impl Model {
	// x[h, img, time1, time2]
	// for all time2 > time1, set to negative infinity
	pub fn add_mask_future_times(&mut self, params: MaskFutureTimesParams) {
		let data_type = params.data_type;
		debug_assert!(self.layers.len() > 0);
		
		let layer_prev = self.layers.len() - 1;
		
		let x_shape = self.layers[layer_prev].y.ravel_time_shape(); // (input to this layer is the output of the previous layer)
		
		debug_assert!(x_shape.n != 1 && x_shape.c != 1 && x_shape.h != 1 && x_shape.w != 1);
		debug_assert!(x_shape.h == x_shape.w); // this should be the time dimension
		
		self.layers.push( Layer::new(
			vec![layer_prev],
			InternalTypes::MaskFutureTimes(MaskFutureTimesInternals {params}),
			Tensor::new(data_type, x_shape),
			String::from("MaskFutureTimes"),
			data_type
		));
	}
	
	pub fn load_mask_future_times(&mut self, layer_keys: &Vec<KeyPair>) {
		self.add_mask_future_times(MaskFutureTimesParams {
			scale: find_req_key_parse("scale", layer_keys),
			data_type: find_req_key_parse("data_type", layer_keys)
		});
	}
}
