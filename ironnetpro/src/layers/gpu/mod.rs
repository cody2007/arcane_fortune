use std::os::raw::{c_int};
use std::mem::size_of;
pub use crate::cudnn::*;
pub use crate::optimizers::*;
use crate::rand::XorState;
use crate::saving::SvStruct;
use crate::model::Model;
use crate::data_wrappers::*;
use crate::string_formatting::num_format;
#[cfg(feature="profile")]
pub use crate::profiling_internal::Guard;

//pub const NAN_PROP: cudnnNanPropagation_t = cudnnNanPropagation_t::CUDNN_PROPAGATE_NAN;
pub const NAN_PROP: cudnnNanPropagation_t = cudnnNanPropagation_t::CUDNN_NOT_PROPAGATE_NAN;

////////////////////////////////////////// primitive layers
pub mod img; pub use img::*;
pub mod time_series; pub use time_series::*;
pub mod conv; pub use conv::*;
pub mod pooling; pub use pooling::*;
pub mod softmax; pub use softmax::*;
pub mod activation; pub use activation::*;
pub mod add; pub use add::*;
pub mod mul; pub use mul::*;
pub mod sum_reduce; pub use sum_reduce::*;
pub mod pow; pub use pow::*;
pub mod lstm; pub use lstm::*;
pub mod scale; pub use scale::*;
pub mod transpose_reshape; pub use transpose_reshape::*;
pub mod bias; pub use bias::*;
pub mod bias_channels_custom; pub use bias_channels_custom::*;
pub mod fully_connected; pub use fully_connected::*;
pub mod fully_connected_w_bias; pub use fully_connected_w_bias::*;
pub mod fully_connected_w_bias_relu; pub use fully_connected_w_bias_relu::*;
pub mod elementwise_affine; pub use elementwise_affine::*;

////// multi-head attn components
pub mod QKV; pub use QKV::*; // layers to create queries, keys, and values for multi-head-attn (transformers)
pub mod mul_Q_K; pub use mul_Q_K::*; // multiplies queries and keys
pub mod mul_Q_and_pos; pub use mul_Q_and_pos::*; // multiplies queries with learned relative position encodings
pub mod mul_softmaxQK_and_V; pub use mul_softmaxQK_and_V::*; // multiplies softmaxQK * V, where softmaxQK is the output of mul_Q_K

pub mod mask_future_times; pub use mask_future_times::*;
pub mod mask_future_times_add; pub use mask_future_times_add::*; // (1) adds the Q*K + Q*pos layers, (2) scales, (3) mask future times
pub mod mask_future_times_add_softmax_w; pub use mask_future_times_add_softmax_w::*;
	// ^ (1) adds the Q*K + Q*pos layers, (2) scales, (3) mask future times, (4) softmax across w dimension
pub mod QK_plus_Qpos_mask_future_times_softmax_w; pub use QK_plus_Qpos_mask_future_times_softmax_w::*;
	//  ^
	//	1. QK 	(output shape: y[h,img,time1,time2], inputs both: x[h,img,time1,vec_out)
	//	2. Q*pos 	(output shape: ", pos: x[h,vec_out, time2]
	// 	3. Q*K + Q*pos 	(both of shape: x]h,img,time1,time2]
	//	4. mask future time: for all time2 > time1, set to negative infinity
	//	5. scale outputs
	//	6. softmax across w dimension (time2)

pub mod QK_plus_Qpos_mask_future_times_softmaxw_mul_V; pub use QK_plus_Qpos_mask_future_times_softmaxw_mul_V::*;
	//  ^
	//	1. QK 	(output shape: y[h,img,time1,time2], inputs both: x[h,img,time1,vec_out)
	//	2. Q*pos 	(output shape: ", pos: x[h,vec_out, time2]
	// 	3. Q*K + Q*pos 	(both of shape: x]h,img,time1,time2]
	//	4. mask future time: for all time2 > time1, set to negative infinity
	//	5. scale outputs
	//	6. QKP = softmax across w dimension (time2)
	//	7. QKP * V

//////////////////////////////////////////////// derived from primitive layers
pub mod softmax_cross_entropy_loss; pub use softmax_cross_entropy_loss::*;
pub mod correlation_loss; pub use correlation_loss::*;
pub mod least_square_loss; pub use least_square_loss::*;
pub mod multi_head_attn; pub use multi_head_attn::*;
pub mod layer_norm; pub use layer_norm::*;

pub enum InternalTypes {
	Conv(ConvInternals),
	Pooling(PoolingInternals),
	Softmax(SoftmaxInternals),
	Activation(ActivationInternals),
	Add(AddInternals),
	Mul(MulInternals),
	SumReduce(SumReduceInternals),
	Pow(PowInternals),
	LSTM(LSTMInternals),
	QKV(QKVInternals),
	QKPlusQPosMaskFutureTimesSoftmaxW(QKPlusQPosMaskFutureTimesSoftmaxWInternals),
	QKPlusQPosMaskFutureTimesSoftmaxWMulV(QKPlusQPosMaskFutureTimesSoftmaxWMulVInternals),
	
	MulQK(MulQKInternals),
	MulQAndPos(MulQAndPosInternals),
	MulSoftmaxQKAndV(MulSoftmaxQKAndVInternals),
	MaskFutureTimes(MaskFutureTimesInternals),
	
	MaskFutureTimesAdd(MaskFutureTimesAddInternals),
	MaskFutureTimesAddSoftmaxW(MaskFutureTimesAddSoftmaxWInternals),
	
	Scale(ScaleInternals),
	TransposeReshape(TransposeReshapeInternals),
	Bias(BiasInternals),
	BiasChannelsCustom(BiasChannelsCustomInternals),
	FullyConnected(FullyConnectedInternals),
	FullyConnectedWBias(FullyConnectedWBiasInternals),
	FullyConnectedWBiasRelu(FullyConnectedWBiasReluInternals),
	ElementwiseAffine(ElementwiseAffineInternals),
	Img(ImgInternals),
	TimeSeries(TimeSeriesInternals)
}

// ex. run_internal!{layer.internals => forward(layer, self) }
macro_rules! run_internal{($internals: expr => $fn: ident ($($args:expr),* )) => {
	match &$internals {
		InternalTypes::Conv(internals) => {internals.$fn($($args),*);}
		InternalTypes::Pooling(internals) => {internals.$fn($($args),*);}
		InternalTypes::Softmax(internals) => {internals.$fn($($args),*);}
		InternalTypes::Activation(internals) => {internals.$fn($($args),*);}
		InternalTypes::Add(internals) => {internals.$fn($($args),*);}
		InternalTypes::Mul(internals) => {internals.$fn($($args),*);}
		InternalTypes::SumReduce(internals) => {internals.$fn($($args),*);}
		InternalTypes::Pow(internals) => {internals.$fn($($args),*);}
		InternalTypes::LSTM(internals) => {internals.$fn($($args),*);}
		InternalTypes::QKV(internals) => {internals.$fn($($args),*);}
		InternalTypes::QKPlusQPosMaskFutureTimesSoftmaxW(internals) => {internals.$fn($($args),*);}
		InternalTypes::QKPlusQPosMaskFutureTimesSoftmaxWMulV(internals) => {internals.$fn($($args),*);}
		
		InternalTypes::MulQK(internals) => {internals.$fn($($args),*);}
		InternalTypes::MulQAndPos(internals) => {internals.$fn($($args),*);}
		InternalTypes::MulSoftmaxQKAndV(internals) => {internals.$fn($($args),*);}
		InternalTypes::MaskFutureTimes(internals) => {internals.$fn($($args),*);}
		
		InternalTypes::MaskFutureTimesAdd(internals) => {internals.$fn($($args),*);}
		InternalTypes::MaskFutureTimesAddSoftmaxW(internals) => {internals.$fn($($args),*);}
		
		InternalTypes::Scale(internals) => {internals.$fn($($args),*);}
		InternalTypes::TransposeReshape(internals) => {internals.$fn($($args),*);}
		InternalTypes::Bias(internals) => {internals.$fn($($args),*);}
		InternalTypes::BiasChannelsCustom(internals) => {internals.$fn($($args),*);}
		InternalTypes::FullyConnected(internals) => {internals.$fn($($args),*);}
		InternalTypes::FullyConnectedWBias(internals) => {internals.$fn($($args),*);}
		InternalTypes::FullyConnectedWBiasRelu(internals) => {internals.$fn($($args),*);}
		InternalTypes::ElementwiseAffine(internals) => {internals.$fn($($args),*);}
		
		InternalTypes::Img(internals) => {internals.$fn($($args),*);}
		InternalTypes::TimeSeries(internals) => {internals.$fn($($args),*)}
	}
};}

// ex. run_composite!{layer.internals => rms_descent(gradients(), ...)}
macro_rules! run_composite{($internals: expr => $fn_outer: ident ( $fn: ident ($($args:expr),* ), $($args_outer:expr),*)) => {
	$fn_outer(match &$internals {
		InternalTypes::Conv(internals) => {internals.$fn($($args),*)}
		InternalTypes::Pooling(internals) => {internals.$fn($($args),*)}
		InternalTypes::Softmax(internals) => {internals.$fn($($args),*)}
		InternalTypes::Activation(internals) => {internals.$fn($($args),*)}
		InternalTypes::Add(internals) => {internals.$fn($($args),*)}
		InternalTypes::Mul(internals) => {internals.$fn($($args),*)}
		InternalTypes::SumReduce(internals) => {internals.$fn($($args),*)}
		InternalTypes::Pow(internals) => {internals.$fn($($args),*)}
		InternalTypes::LSTM(internals) => {internals.$fn($($args),*)}
		InternalTypes::QKV(internals) => {internals.$fn($($args),*)}
		InternalTypes::QKPlusQPosMaskFutureTimesSoftmaxW(internals) => {internals.$fn($($args),*)}
		InternalTypes::QKPlusQPosMaskFutureTimesSoftmaxWMulV(internals) => {internals.$fn($($args),*)}
		
		InternalTypes::MulQK(internals) => {internals.$fn($($args),*)}
		InternalTypes::MulQAndPos(internals) => {internals.$fn($($args),*)}
		InternalTypes::MulSoftmaxQKAndV(internals) => {internals.$fn($($args),*)}
		InternalTypes::MaskFutureTimes(internals) => {internals.$fn($($args),*)}
		
		InternalTypes::MaskFutureTimesAdd(internals) => {internals.$fn($($args),*)}
		InternalTypes::MaskFutureTimesAddSoftmaxW(internals) => {internals.$fn($($args),*)}
		
		InternalTypes::Scale(internals) => {internals.$fn($($args),*)}
		InternalTypes::TransposeReshape(internals) => {internals.$fn($($args),*)}
		InternalTypes::Bias(internals) => {internals.$fn($($args),*)}
		InternalTypes::BiasChannelsCustom(internals) => {internals.$fn($($args),*)}
		InternalTypes::FullyConnected(internals) => {internals.$fn($($args),*)}
		InternalTypes::FullyConnectedWBias(internals) => {internals.$fn($($args),*)}
		InternalTypes::FullyConnectedWBiasRelu(internals) => {internals.$fn($($args),*)}
		InternalTypes::ElementwiseAffine(internals) => {internals.$fn($($args),*)}
		
		InternalTypes::Img(internals) => {internals.$fn($($args),*)}
		InternalTypes::TimeSeries(internals) => {internals.$fn($($args),*)}
	}, $($args_outer),*);
};}

macro_rules! run_internal_ret_val{($internals: expr => $fn: ident ($($args:expr),* ) => $ret_val: expr) => {
	$ret_val = match &$internals {
		InternalTypes::Conv(internals) => {internals.$fn($($args),*)}
		InternalTypes::Pooling(internals) => {internals.$fn($($args),*)}
		InternalTypes::Softmax(internals) => {internals.$fn($($args),*)}
		InternalTypes::Activation(internals) => {internals.$fn($($args),*)}
		InternalTypes::Add(internals) => {internals.$fn($($args),*)}
		InternalTypes::Mul(internals) => {internals.$fn($($args),*)}
		InternalTypes::SumReduce(internals) => {internals.$fn($($args),*)}
		InternalTypes::Pow(internals) => {internals.$fn($($args),*)}
		InternalTypes::LSTM(internals) => {internals.$fn($($args),*)}
		InternalTypes::QKV(internals) => {internals.$fn($($args),*)}
		InternalTypes::QKPlusQPosMaskFutureTimesSoftmaxW(internals) => {internals.$fn($($args),*)}
		InternalTypes::QKPlusQPosMaskFutureTimesSoftmaxWMulV(internals) => {internals.$fn($($args),*)}
		
		InternalTypes::MulQK(internals) => {internals.$fn($($args),*)}
		InternalTypes::MulQAndPos(internals) => {internals.$fn($($args),*)}
		InternalTypes::MulSoftmaxQKAndV(internals) => {internals.$fn($($args),*)}
		InternalTypes::MaskFutureTimes(internals) => {internals.$fn($($args),*)}
		
		InternalTypes::MaskFutureTimesAdd(internals) => {internals.$fn($($args),*)}
		InternalTypes::MaskFutureTimesAddSoftmaxW(internals) => {internals.$fn($($args),*)}
		
		InternalTypes::Scale(internals) => {internals.$fn($($args),*)}
		InternalTypes::TransposeReshape(internals) => {internals.$fn($($args),*)}
		InternalTypes::Bias(internals) => {internals.$fn($($args),*)}
		InternalTypes::BiasChannelsCustom(internals) => {internals.$fn($($args),*)}
		InternalTypes::FullyConnected(internals) => {internals.$fn($($args),*)}
		InternalTypes::FullyConnectedWBias(internals) => {internals.$fn($($args),*)}
		InternalTypes::FullyConnectedWBiasRelu(internals) => {internals.$fn($($args),*)}
		InternalTypes::ElementwiseAffine(internals) => {internals.$fn($($args),*)}
		
		InternalTypes::Img(internals) => {internals.$fn($($args),*)}
		InternalTypes::TimeSeries(internals) => {internals.$fn($($args),*)}
	};
};}

pub enum Output {
	Tensor(Tensor),
	RNNData(RNNData)
}

macro_rules! run_output{($output: expr => $fn: ident ($($args:expr),* )) => {
	match $output {
		Output::Tensor(tensor) => {tensor.$fn($($args),*);}
		Output::RNNData(rnn_data) => {rnn_data.$fn($($args),*);}
	}
};}

impl Output {
	impl_output_common!(Tensor, RNNData);
	
	pub fn zero_out(&self) {
		run_output!(self => zero_out());
	}
	
	pub fn one_out(&self) {
		run_output!(self => one_out());
	}
	
	pub fn set<T>(&self, src: &Vec<T>) {
		run_output!(self => set(src));
	}
	
	pub fn ret(&self) -> Vec<f32> {
		match self {
			Self::Tensor(tensor) => {tensor.ret()}
			Self::RNNData(rnn_data) => {rnn_data.ret()}
		}
	}
	
	pub fn n_elements(&self) -> size_t {
		match self {
			Self::Tensor(tensor) => {tensor.shape.n_elements()}
			Self::RNNData(rnn_data) => {rnn_data.n_elements()}
		}
	}
	
	pub fn ravel_time_tensor_desc(&self) -> cudnnTensorDescriptor_t {
		match self {
			Self::Tensor(tensor) => {tensor.desc.val}
			Self::RNNData(rnn_data) => {rnn_data.ravel_time_tensor_desc.val}
		}
	}
	
	pub fn mem(&self) -> gpuMem_t {
		match self {
			Self::Tensor(tensor) => {tensor.mem.val}
			Self::RNNData(rnn_data) => {rnn_data.mem.val}
		}
	}
	
	pub fn data_type(&self) -> cudnnDataType_t {
		match self {
			Self::Tensor(tensor) => {tensor.mem.dataType}
			Self::RNNData(rnn_data) => {rnn_data.mem.dataType}
		}
	}
}

pub struct WeightsAdam {
	pub m: gpuMem,
	pub v: gpuMem
}

pub struct Layer {
	pub x_layers: Vec<usize>, // inputs (layer inds)
	pub internals: InternalTypes,
	
	pub y: Output,
	pub dy: Output,
	
	pub data_type: cudnnDataType_t,
	pub nm: String,
	
	pub weights_rms_tmp: Option<gpuMem>, // basically the momentum for the weight derivatives
	
	pub weights_adam: Option<WeightsAdam>,
	
	pub run_fwd: bool
}

macro_rules! print_layer_sz{($internals: expr, $y: expr, $nm: expr) => {
	let weights_vec;
	run_internal_ret_val!($internals => gradients() => weights_vec);
	let mut weights_sz = 0;
	for weights in weights_vec.iter() {
		weights_sz += weights.len*2*size_of::<f32>();
	}
	
	let workspace_sz;
	run_internal_ret_val!($internals => workspace_sz() => workspace_sz);
	
	println!("{}: (weights: {} workspace: {})  {}", num_format(($y.mem.bytes*2) + weights_sz + workspace_sz), 
			num_format(weights_sz), num_format(workspace_sz), $nm);
}}

impl Layer {
	// set layer output
	pub fn set_output<T>(&self, vals: &Vec<T>) {
		if let Output::Tensor(tensor) = &self.y {
			//tensor.mem.set(vals);
			tensor.mem.set_underfilled(vals);
		}else{
			panic!("output is not a tensor. possible sequence?");
		}
	}
	
	// updates descriptor for changes in sequence lengths for each batch
	pub fn set_output_seq<T>(&mut self, vals: &Vec<T>, seq_len_array: &Vec<c_int>) {
		let max_seq_len = *seq_len_array.iter().max().unwrap();
		debug_assert!((vals.len() % max_seq_len as usize) == 0);
		
		match &mut self.y {
			Output::RNNData(y) => {
				debug_assert!((max_seq_len as usize *
							seq_len_array.len() *
							y.vec_sz as usize) == vals.len());

				y.update_valid_tpoints(seq_len_array, max_seq_len);
				y.mem.set_underfilled(vals);
				self.dy.rnn_data_mut().update_valid_tpoints(seq_len_array, max_seq_len);
			} Output::Tensor(y) => {
				let actual_batch_sz = seq_len_array.len() as c_int;
				let batch_sz = max_seq_len * actual_batch_sz;
				// ^ batch tensor dim wraps time (max_seq_len) and actual batch size
				y.update_batch_sz(batch_sz);
				y.mem.set_underfilled(vals);
				self.dy.tensor_mut().update_batch_sz(batch_sz);
			}
		}
	}

	pub fn new(x_layers: Vec<usize>, internals: InternalTypes, 
			y: Tensor, nm: String, data_type: cudnnDataType_t) -> Self {
		
		print_layer_sz!(internals, y, nm);
		
		Self {
			x_layers,
			internals,
			dy: Output::Tensor(Tensor::zeros(data_type, y.shape)),
			y: Output::Tensor(y),
			data_type,
			nm: nm.clone(),
			weights_rms_tmp: None,
			weights_adam: None,
			run_fwd: false
		}
		
		//println!("{}: {}", (used_bytes() as f32) / (1024.*1024.), nm);
	}
	
	pub fn new_time_series(x_layers: Vec<usize>, internals: InternalTypes,
			y: RNNData, nm: String, data_type: cudnnDataType_t) -> Self {
		
		print_layer_sz!(internals, y, nm);

		Self {
			x_layers,
			internals,
			dy: Output::RNNData(RNNData::zeros(data_type, y.max_seq_len, y.batch_sz,
						y.vec_sz, &y.seq_len_array)),
			y: Output::RNNData(y),
			data_type,
			nm,
			weights_rms_tmp: None,
			weights_adam: None,
			run_fwd: false
		}
	}
	
	pub fn new_w_dy(x_layers: Vec<usize>, internals: InternalTypes, 
			y: Tensor, dy: Tensor, nm: String, data_type: cudnnDataType_t) -> Self {
		
		print_layer_sz!(internals, y, nm);
		
		Self {
			x_layers,
			internals,
			dy: Output::Tensor(dy),
			y: Output::Tensor(y),
			data_type,
			nm,
			weights_rms_tmp: None,
			weights_adam: None,
			run_fwd: false
		}
	}
	
	// uses output of prediction layer to construct and return the next inputs   (zeroth return variable)
	// also returns the probability for the chosen action				     (first return variable: shape: [batch_sz])
	pub fn recursive_net_input_frm_output(&self, t: usize, batch_sz: usize, dict_sz: usize, rng: &mut XorState) -> (Vec<f32>, Vec<f32>) {
		// use prior output as input
		if t != 0 {
			//////// take max of output across input_sz dim [batch_sz, input_sz]
			let mut y_input = self.y.ret();
			let mut probs_chosen = Vec::with_capacity(batch_sz);
			debug_assert!(y_input.len() == (batch_sz*dict_sz));
			
			// find max across each image, then set max = 1 and all other values to 0
			for img in 0..batch_sz {
				// probabilstically sample
				let max_ind = {
					let prob_val = rng.gen_f32b();
					let mut max_ind = 0;
					let mut val_sum = 0.;
					
					for ind in 0..dict_sz {
						let ind_use = img*dict_sz + ind;
						val_sum += y_input[ind_use].exp();
						if prob_val < val_sum {
							max_ind = ind_use;
							probs_chosen.push(prob_val);
							break;
						}
					}
					max_ind
				};
				// max_ind indexes pred
				
				// set vals
				for (ind, yv) in y_input.iter_mut().enumerate().skip(img*dict_sz).take(dict_sz) {
					*yv = if ind != max_ind {0.} else {1.};
				}
			}
			
			debug_assert!(probs_chosen.len() == batch_sz);
			(y_input, probs_chosen)
			
		// first input in sequence (nulls)
		}else{
			(vec!{0.; batch_sz*dict_sz}, vec!{0.; batch_sz})
		}
	}
}

pub trait Run {
	fn forward(&self, _: &Layer, _: &Model);
	fn backward(&self, _: &Layer, _: &Model);
	fn sv_arch(&self, txt: &mut String);
	
	////////////////////////////////////////////////////////
	// default empty functions:
	/////////////////////////////////
	
	fn gradients(&self) -> Vec<Weights> {Vec::new()}
	
	fn workspace_sz(&self) -> usize {0}
	
	/////
	// rnns:
	fn zero_out_internal_gradients(&self) {}
	fn zero_out_internal_states(&self) {}
	
	fn ret_internal_states(&self) -> Option<Vec<Vec<f32>>> {None} // [variable][vals]
	fn set_internal_states(&self, _states: &Vec<Vec<f32>>) {}
	fn remap_internal_states(&self, _reorder_imgs: &Vec<usize>, _batch_sz: usize) {}
	
	///////
	// save/load
	fn sv_weights(&self, _: &Layer, _: &Model, _save_dir: &str, _file_nm: &str) {}
	fn sv_gradients(&self, _save_dir: &str, _file_nm: &str) {}
	
	fn ld_weights(&self, _: &Layer, _: &Model, _save_dir: &str, _file_nm: &str) {}
}

