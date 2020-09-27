use super::*;
use crate::rand::XorState;
use crate::saving::{KeyPair, find_req_key_parse};
use crate::layers::ConvParams;

pub struct ConvInternals {
	params: ConvParams,
	
	conv_desc: ConvolutionDescriptor,
	pub filter: Filter,
	dfilter: Filter,
	
	workspace_fwd: WorkspaceConvolutionFwd,
	workspace_bwd_data: WorkspaceConvolutionBwdData,
	workspace_bwd_filter: WorkspaceConvolutionBwdFilter
}

const MODE: cudnnConvolutionMode_t = cudnnConvolutionMode_t::CUDNN_CONVOLUTION;

const PREF_FWD_ALG: cudnnConvolutionFwdPreference_t = cudnnConvolutionFwdPreference_t::CUDNN_CONVOLUTION_FWD_PREFER_FASTEST;
const PREF_BWD_DATA_ALG: cudnnConvolutionBwdDataPreference_t = cudnnConvolutionBwdDataPreference_t::CUDNN_CONVOLUTION_BWD_DATA_PREFER_FASTEST;
const PREF_BWD_FILTER_ALG: cudnnConvolutionBwdFilterPreference_t = cudnnConvolutionBwdFilterPreference_t::CUDNN_CONVOLUTION_BWD_FILTER_PREFER_FASTEST;

const SZ_PREF_FWD_ALG: size_t = 0;
const SZ_PREF_BWD_DATA_ALG: size_t = 0;
const SZ_PREF_BWD_FILTER_ALG: size_t = 0;

impl Run for ConvInternals {
	fn forward(&self, layer: &Layer, model: &Model) {
		let x = &model.layers[layer.x_layers[0]].y; // output of input layer is the input for this layer
		let y = &layer.y;
		
		unsafe {raw::cudnnConvolutionForward(
				model.handle.cudnn_val,
				
				model.one(layer.data_type),
				x.ravel_time_tensor_desc(), x.mem(),
				self.filter.desc.val, self.filter.mem.val,
				
				self.conv_desc.val,
				self.workspace_fwd.alg,
				self.workspace_fwd.mem.val,
				self.workspace_fwd.mem.bytes,
				
				model.zero(layer.data_type),
				y.ravel_time_tensor_desc(), y.mem())}.chk_err();
	}
	
	fn backward(&self, layer: &Layer, model: &Model) {
		let in_layer = &model.layers[layer.x_layers[0]]; // output of input layer is the input for this layer
		
		let dx = &in_layer.dy; 
		let x = &in_layer.y;
		let dy = &layer.dy;
		
		unsafe {raw::cudnnConvolutionBackwardData(
				model.handle.cudnn_val,
				
				model.one(layer.data_type),
				self.filter.desc.val, self.filter.mem.val,
				dy.ravel_time_tensor_desc(), dy.mem(),
				
				self.conv_desc.val,
				self.workspace_bwd_data.alg,
				self.workspace_bwd_data.mem.val,
				self.workspace_bwd_data.mem.bytes,
				
				model.one(layer.data_type),
				dx.ravel_time_tensor_desc(), dx.mem())}.chk_err();
		
		unsafe {raw::cudnnConvolutionBackwardFilter(
				model.handle.cudnn_val,
				
				model.one(layer.data_type),
				x.ravel_time_tensor_desc(), x.mem(),
				dy.ravel_time_tensor_desc(), dy.mem(),
				
				self.conv_desc.val,
				self.workspace_bwd_filter.alg,
				self.workspace_bwd_filter.mem.val,
				self.workspace_bwd_filter.mem.bytes,
				
				model.one(layer.data_type),
				self.dfilter.desc.val, self.dfilter.mem.val)}.chk_err();
	}
	
	fn zero_out_internal_gradients(&self) {
		self.dfilter.zero_out();
	}
	
	fn gradients(&self) -> Vec<Weights> {
		vec![Weights {
			w_desc: self.filter.tensor_desc.val, 
			w_mem: self.filter.mem.val,
			dw_desc: self.dfilter.tensor_desc.val,
			dw_mem: self.dfilter.mem.val,
			len: self.filter.mem.n_elements,
			data_type: self.filter.mem.dataType
		}]
	}
	
	fn sv_arch(&self, txt: &mut String) {
		txt.push_str(&format!("\tn_filters: {}\n", self.params.n_filters));
		txt.push_str(&format!("\tfilter_sz_h: {}\n", self.params.filter_sz_h));
		txt.push_str(&format!("\tfilter_sz_w: {}\n", self.params.filter_sz_w));
		txt.push_str(&format!("\tpad_h: {}\n", self.params.pad_h));
		txt.push_str(&format!("\tpad_w: {}\n", self.params.pad_w));
		txt.push_str(&format!("\tstride: {}\n", self.params.stride));
		txt.push_str(&format!("\tnorm_scale: {}\n", self.params.norm_scale));
		txt.push_str(&format!("\tdata_type: {}\n", self.params.data_type));
	}
	
	fn sv_weights(&self, _: &Layer, _: &Model, save_dir: &str, file_nm: &str) {
		self.filter.sv(save_dir, file_nm);
	}
	
	fn sv_gradients(&self, save_dir: &str, file_nm: &str) {
		self.dfilter.sv(save_dir, file_nm);
	}
	
	fn ld_weights(&self, _: &Layer, _: &Model, save_dir: &str, file_nm: &str) {
		self.filter.ld(save_dir, file_nm);
	}
	
	fn workspace_sz(&self) -> usize {
		self.workspace_fwd.mem.bytes +
		self.workspace_bwd_data.mem.bytes +
		self.workspace_bwd_filter.mem.bytes
	}
}

impl Model {
	pub fn add_conv(&mut self, params: ConvParams, rng: &mut XorState) {
		let data_type = params.data_type;
		debug_assert!(self.layers.len() > 0);
		
		let layer_prev_ind = self.layers.len() - 1;
		let layer_prev = &self.layers[layer_prev_ind];
		
		let x = &layer_prev.y; // (input to this layer is the output of the previous layer)
		let dx = &layer_prev.dy;
		
		let x_shape = x.ravel_time_shape();
		let n_input_channels = x_shape.c;
		let y_shape = params.out_shape(&x_shape);
		
		let filter_shape = FilterShape {
				k: params.n_filters,
				c: n_input_channels,
				h: params.filter_sz_h,
				w: params.filter_sz_w
		};
		
		let y = Output::Tensor(Tensor::new(data_type, y_shape));
		let dy = Output::Tensor(Tensor::zeros(data_type, y_shape));
		
		/*let (y, dy) = match x {
			Output::Tensor(_) => {
				let y = Tensor::new(data_type, y_shape);
				let dy = Tensor::zeros(data_type, y_shape);
				
				(Output::Tensor(y), Output::Tensor(dy))
			} Output::RNNData(x_rnn_data) => {
				debug_assert!(pad_h == 0 && pad_w == 0 && stride == 1); // output should be same size
				debug_assert!(h_out == 1 && w_out == 1, "h_out {} w_out {}", h_out, w_out);
				
				let max_seq_len = x_rnn_data.max_seq_len;
				let batch_sz = x_rnn_data.batch_sz;
				let vec_sz = n_filters;
				let seq_len_array = &x_rnn_data.seq_len_array;
				
				let y = RNNData::new(data_type, max_seq_len, batch_sz, vec_sz, seq_len_array);
				let dy = RNNData::zeros(data_type, max_seq_len, batch_sz, vec_sz, seq_len_array);
				
				println!("h_out {} w_out {} n_input_channels {} n_imgs {}", h_out, w_out, n_input_channels, n_imgs);
				println!("max_seq_len {} batch_sz {} vec_sz {} seq_len_array {}", max_seq_len, batch_sz, vec_sz, seq_len_array.len());
				
				(Output::RNNData(y), Output::RNNData(dy))
			}
		};*/
		
		let filter = Filter::new_norm(data_type, filter_shape, params.norm_scale, rng);
		let dfilter = Filter::zeros(data_type, filter_shape);
		
		let conv_desc = ConvolutionDescriptor::new(params.pad_h, params.pad_w, 
						params.stride, params.stride, 1, 1, MODE, data_type);
		
		let workspace_fwd = self.handle.allocate_workspace_convolution_fwd_pref(x, &filter, &conv_desc, &y, PREF_FWD_ALG, SZ_PREF_FWD_ALG);
		let workspace_bwd_data = self.handle.allocate_workspace_convolution_bwd_data_pref(&filter, &dy, &conv_desc, &dx, PREF_BWD_DATA_ALG, SZ_PREF_BWD_DATA_ALG);
		let workspace_bwd_filter = self.handle.allocate_workspace_convolution_bwd_filter_pref(x, &dy, &conv_desc, &dfilter, PREF_BWD_FILTER_ALG, SZ_PREF_BWD_FILTER_ALG);
		
		println!("conv filter: {}", filter.shape.to_string());
		
		match y {
			Output::Tensor(y) => {
				self.layers.push( 
					Layer::new(vec!{layer_prev_ind; 1},
						InternalTypes::Conv(ConvInternals {
							params,
							conv_desc,
							filter, 
							dfilter,
							workspace_fwd,
							workspace_bwd_data,
							workspace_bwd_filter
						}),
					y,
					String::from("conv"),
					data_type
				));
			} Output::RNNData(y) => {
				self.layers.push( 
					Layer::new_time_series(vec!{layer_prev_ind; 1},
						InternalTypes::Conv(ConvInternals {
							params,
							conv_desc,
							filter, 
							dfilter,
							workspace_fwd,
							workspace_bwd_data,
							workspace_bwd_filter
						}),
					y,
					String::from("conv"),
					data_type
				));
			}
		}
	}
	
	pub fn load_conv(&mut self, layer_keys: &Vec<KeyPair>, rng: &mut XorState) {
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

