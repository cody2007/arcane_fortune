use super::*;
use crate::layers::MaskFutureTimesParams;
use crate::saving::{KeyPair, find_req_key_parse};
use std::os::raw::c_void;

// Inputs both of shape: x[h, img, time1, time2]
// 	1. add previous layer to layer2_ind
//	2. mask future time: for all time2 > time1, set to negative infinity
//	3. scale outputs
//	4. softmax across w dimension (time2)

const SOFTMAX_ALG: cudnnSoftmaxAlgorithm_t = cudnnSoftmaxAlgorithm_t::CUDNN_SOFTMAX_ACCURATE;
const MODE: cudnnSoftmaxMode_t = cudnnSoftmaxMode_t::CUDNN_SOFTMAX_MODE_INSTANCE; // compute across all values for each img

pub struct MaskFutureTimesAddSoftmaxWInternals {
	params: MaskFutureTimesParams,
	softmax_desc: TensorDescriptor
}

impl Run for MaskFutureTimesAddSoftmaxWInternals {
	fn forward(&self, layer: &Layer, model: &Model) {
		#[cfg(feature="profile")]
		let _g = Guard::new("mask future times add fwd");
		
		debug_assert!(layer.x_layers.len() == 2);
		
		let x1 = &model.layers[layer.x_layers[0]].y.tensor();
		let x2 = &model.layers[layer.x_layers[1]].y.tensor();
		let y = &layer.y.tensor();
		
		let n_time = x1.shape.w;
		debug_assert!(x1.shape.h == n_time);
		debug_assert!(x1.shape == x2.shape);
		
		let n_exemplars = x1.shape.n * x1.shape.c;
		
		debug_assert!(self.params.data_type == cudnnDataType_t::CUDNN_DATA_FLOAT &&
				  x1.mem.dataType == cudnnDataType_t::CUDNN_DATA_FLOAT &&
				  x2.mem.dataType == x1.mem.dataType);
		
		let y_tmp = model.shared_workspace.as_ref().unwrap();
		
		unsafe {raw::mask_future_times_add(y_tmp.val, x1.mem.val, x2.mem.val, self.params.scale,
				n_exemplars as size_t, n_time as size_t)};
		
		unsafe {raw::cudnnSoftmaxForward(
					model.handle.cudnn_val, SOFTMAX_ALG, MODE,
					
					model.one(layer.data_type),
					self.softmax_desc.val, y_tmp.val,
					
					model.zero(layer.data_type),
					self.softmax_desc.val, y.mem.val)}.chk_err();
	}
	
	fn backward(&self, layer: &Layer, model: &Model) {
		#[cfg(feature="profile")]
		let _g = Guard::new("mask future times add bwd");
		
		debug_assert!(layer.x_layers.len() == 2);
		
		let dx1 = &model.layers[layer.x_layers[0]].dy.tensor();
		let dx2 = &model.layers[layer.x_layers[1]].dy.tensor();
		let dy = &layer.dy.tensor();
		let y = &layer.y.tensor();
		
		debug_assert!(self.params.data_type == cudnnDataType_t::CUDNN_DATA_FLOAT &&
				  dx1.mem.dataType == cudnnDataType_t::CUDNN_DATA_FLOAT &&
				  dx2.mem.dataType == dx1.mem.dataType);
		// the following code doesn't strictly require that the data are floats, only that dx & dy are the same datatype
		
		// in place softmax backward
		unsafe {raw::cudnnSoftmaxBackward(
					model.handle.cudnn_val, SOFTMAX_ALG, MODE,
					
					model.one(layer.data_type),
					self.softmax_desc.val, y.mem(),
					self.softmax_desc.val, dy.mem(),
					
					model.zero(layer.data_type),
					self.softmax_desc.val, dy.mem())}.chk_err();
		
		// add result to each dx1 & dx2
		unsafe {cudnnAddTensor(model.handle.cudnn_val,
					vec![self.params.scale].as_ptr() as *const c_void,
					dy.desc.val, dy.mem.val,
					
					model.one(layer.data_type),
					dx1.desc.val, dx1.mem.val)}.chk_err();
		
		unsafe {cudnnAddTensor(model.handle.cudnn_val,
					vec![self.params.scale].as_ptr() as *const c_void,
					dy.desc.val, dy.mem.val,
					
					model.one(layer.data_type),
					dx2.desc.val, dx2.mem.val)}.chk_err();

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
	pub fn add_mask_future_times_add_softmax_w(&mut self, layer2_ind: usize, params: MaskFutureTimesParams) {
		let data_type = params.data_type;
		debug_assert!(self.layers.len() > 0);
		
		let layer_prev = self.layers.len() - 1;
		
		let x = self.layers[layer_prev].y.tensor();
		let x_shape = x.shape;
		
		debug_assert!(x.shape.n != 1 && x.shape.c != 1 && x.shape.h != 1 && x.shape.w != 1);
		assert!(x.shape.h == x.shape.w); // this should be the time dimension
		assert!(self.layers[layer2_ind].y.ravel_time_shape() == x.shape);
		
		let softmax_desc = TensorDescriptor::new(params.data_type,
						TENSOR_FORMAT,
						TensorShape {
							n: x.shape.n * x.shape.c * x.shape.h,
							c: 1,
							h: 1,
							w: x.shape.w
						});
		
		self.allocate_shared_workspace(params.data_type, x_shape.n_elements());
		
		self.layers.push( Layer::new(
			vec![layer_prev, layer2_ind],
			InternalTypes::MaskFutureTimesAddSoftmaxW(
				MaskFutureTimesAddSoftmaxWInternals {
						params,
						softmax_desc
				}),
			Tensor::new(data_type, x_shape),
			String::from("MaskFutureTimesAddSoftmaxW"),
			data_type
		));
	}
	
	pub fn load_mask_future_times_add_softmax_w(&mut self, x_layers: &Vec<usize>, layer_keys: &Vec<KeyPair>) {
		assert!(x_layers.len() == 2);
		assert!(x_layers[0] > x_layers[1]); // x_layers[0] should be previous layer
		
		self.add_mask_future_times_add_softmax_w(x_layers[1], MaskFutureTimesParams {
			scale: find_req_key_parse("scale", layer_keys),
			data_type: find_req_key_parse("data_type", layer_keys)
		});
	}
}
