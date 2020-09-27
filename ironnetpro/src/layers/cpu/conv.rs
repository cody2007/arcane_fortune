use super::*;
use crate::rand::XorState;
use crate::saving::{KeyPair, find_req_key_parse};
use crate::layers::ConvParams;

pub struct ConvInternalsCPU {
	params: ConvParams,
	pub filter: FilterCPU,
}

impl RunCPU for ConvInternalsCPU {
	fn forward(&mut self, layer_ind: usize, layers: &mut Vec<LayerCPU>) {
		#[cfg(feature="profile")]
		let _g = Guard::new("conv cpu forward");

		debug_assert!(self.params.pad_h == 0 && self.params.pad_w == 0, "padding not implemented for CPU convolution");
		debug_assert!(self.params.stride == 1, "only strides of 1 are supported for CPU convolution");
		
		let layer = &layers[layer_ind];
		let x = &layers[layer.x_layers[0]].y; // output of input layer is the input for this layer
		let x_shape = x.ravel_time_shape();
		
		let y_shape = self.params.out_shape(&x_shape);
		let mut y = TensorCPU::zeros(y_shape);
		
		let n_imgs = x_shape.n;
		let n_filters = self.params.n_filters;
		let n_channels = x_shape.c;
		let f_h = self.params.filter_sz_h;
		let f_w = self.params.filter_sz_w;
		
		let x_h = x_shape.h;
		let x_w = x_shape.w;
		
		let y_h = y.shape.h;
		let y_w = y.shape.w;
		
		debug_assert!(y.shape.n_elements() == (n_imgs*n_filters*y_h*y_w) as usize);
		debug_assert!(x_shape.n_elements() == (n_imgs*n_channels*x_h*x_w) as usize);
		debug_assert!(self.filter.shape.n_elements() == (n_filters*n_channels*f_h*f_w) as usize);
		
		let xv = x.mem();
		
		for img in 0..n_imgs {
		for f in 0..n_filters {
		for y_i in 0..y_h {
		for y_j in 0..y_w { let yv = &mut y.mem[(img*(n_filters*y_h*y_w) + f*(y_h*y_w) + y_i*y_w + y_j) as usize];
		for f_i in 0..f_h { let x_i = y_i + f_h - f_i - 1; // see: "cuDNN: Efficient Primitives for Deep Learning" - Chetlur, et. al 2014
		for f_j in 0..f_w { let x_j = y_j + f_w - f_j - 1;
			for c in 0..n_channels {
				*yv += // [img, f, y_i, y_j]
					self.filter.mem[(f*(n_channels*f_h*f_w) + c*(f_h*f_w) + f_i*f_w + f_j) as usize] // [f, c, f_i, f_j]
					        * xv[ (img*(n_channels*x_h*x_w) + c*(x_h*x_w) + x_i*x_w + x_j) as usize]; // [img, c, x_i, x_j]
		}}}}}}}
		
		*layers[layer_ind].y.mem_mut() = y.mem;
	}
	
	fn ld_weights(&mut self, save_dir: &str, file_nm: &str) {
		self.filter.ld(save_dir, file_nm);
	}
}

impl ModelCPU {
	pub fn add_conv(&mut self, params: ConvParams, rng: &mut XorState) {
		debug_assert!(self.layers.len() > 0);
		
		let layer_prev_ind = self.layers.len() - 1;
		let layer_prev = &self.layers[layer_prev_ind];
		
		let x = &layer_prev.y; // (input to this layer is the output of the previous layer)
		
		let x_shape = x.ravel_time_shape();
		let y_shape = params.out_shape(&x_shape);
		
		let filter_shape = FilterShape {
				k: params.n_filters,
				c: x_shape.c, // # of input channels
				h: params.filter_sz_h,
				w: params.filter_sz_w
		};
		
		let y = OutputCPU::Tensor(TensorCPU::new(y_shape));
		let filter = FilterCPU::new_norm(filter_shape, params.norm_scale, rng);
		
		match y {
			OutputCPU::Tensor(y) => {
				self.new_layer(vec!{layer_prev_ind; 1},
						InternalTypesCPU::Conv(ConvInternalsCPU {
							params,
							filter, 
						}),
					y,
					String::from("conv")
				);
			} OutputCPU::RNNData(y) => {
				self.new_layer_time_series(vec!{layer_prev_ind; 1},
						InternalTypesCPU::Conv(ConvInternalsCPU {
							params,
							filter,
						}),
					y,
					String::from("conv")
				);
			}
		}
	}
	
	pub fn load_conv(&mut self, layer_keys: &Vec<KeyPair>, rng: &mut XorState) {
		let data_type = find_req_key_parse::<cudnnDataType_t>("data_type", layer_keys);
		debug_assert!(data_type == cudnnDataType_t::CUDNN_DATA_FLOAT);
		
		self.add_conv(ConvParams {
			n_filters: find_req_key_parse("n_filters", layer_keys),
			filter_sz_h: find_req_key_parse("filter_sz_h", layer_keys),
			filter_sz_w: find_req_key_parse("filter_sz_w", layer_keys),
			pad_h: find_req_key_parse("pad_h", layer_keys),
			pad_w: find_req_key_parse("pad_w", layer_keys),
			stride: find_req_key_parse("stride", layer_keys),
			norm_scale: find_req_key_parse("norm_scale", layer_keys),
			data_type: find_req_key_parse("data_type", layer_keys)
		}, rng);
	}
}

