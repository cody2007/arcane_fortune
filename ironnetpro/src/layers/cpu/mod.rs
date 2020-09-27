use std::os::raw::{c_int};
pub use crate::cudnn_common::cudnnDataType_t;
pub use crate::data_wrappers::*;
use crate::model::ModelCPU;
#[cfg(feature="profile")]
use crate::profiling_internal::*;

pub mod img; pub use img::*;
pub mod time_series; pub use time_series::*;
pub mod conv; pub use conv::*;
pub mod softmax; pub use softmax::*;
pub mod mul; pub use mul::*;
pub mod sum_reduce; pub use sum_reduce::*;
pub mod lstm; pub use lstm::*;

pub enum InternalTypesCPU {
	Conv(ConvInternalsCPU),
	Softmax(SoftmaxInternalsCPU),
	Mul(MulInternalsCPU),
	SumReduce(SumReduceInternalsCPU),
	LSTM(LSTMInternalsCPU),
	Img(ImgInternalsCPU),
	TimeSeries(TimeSeriesInternalsCPU)
}

// ex. run_internal_cpu!{layer.internals => forward(layer, self) }
macro_rules! run_internal_cpu{($internals: expr => $fn: ident ($($args:expr),* )) => {
	match $internals {
		InternalTypesCPU::Conv(internals) => {internals.$fn($($args),*);}
		/*InternalTypesCPU::Pooling(internals) => {internals.$fn($($args),*);}*/
		InternalTypesCPU::Softmax(internals) => {internals.$fn($($args),*);}
		/*InternalTypesCPU::Activation(internals) => {internals.$fn($($args),*);}
		InternalTypesCPU::Add(internals) => {internals.$fn($($args),*);}*/
		InternalTypesCPU::Mul(internals) => {internals.$fn($($args),*);}
		InternalTypesCPU::SumReduce(internals) => {internals.$fn($($args),*);}
		//InternalTypesCPU::Pow(internals) => {internals.$fn($($args),*);}
		InternalTypesCPU::LSTM(internals) => {internals.$fn($($args),*);}
		InternalTypesCPU::Img(internals) => {internals.$fn($($args),*);}
		InternalTypesCPU::TimeSeries(internals) => {internals.$fn($($args),*)}
	}
};}

pub enum OutputCPU {
	Tensor(TensorCPU),
	RNNData(RNNDataCPU)
}

impl OutputCPU {
	impl_output_common!(TensorCPU, RNNDataCPU);
	
	pub fn mem(&self) -> &Vec<f32> {
		match self {
			Self::Tensor(tensor) => {&tensor.mem}
			Self::RNNData(rnn_data) => {&rnn_data.mem}
		}
	}

	pub fn mem_mut(&mut self) -> &mut Vec<f32> {
		match self {
			Self::Tensor(tensor) => {&mut tensor.mem}
			Self::RNNData(rnn_data) => {&mut rnn_data.mem}
		}
	}
}

pub struct LayerCPU {
	pub x_layers: Vec<usize>, // inputs (layer inds)
	
	// pub internals: InternalTypesCPU, // (stored in ModelCPU,
	//	because for model.forward() because we need a:
	//    		-mutable reference to layer.y,
	//			-immutable references to layer.y (across model layers for inputs)
	//			-immutable reference to internals
	
	pub y: OutputCPU,
	
	pub nm: String,
	pub run_fwd: bool
}

// new layer functions
impl ModelCPU {
	pub fn new_layer(&mut self, x_layers: Vec<usize>, internals: InternalTypesCPU, 
			y: TensorCPU, nm: String) {	
		self.layers.push(LayerCPU {
			x_layers,
			y: OutputCPU::Tensor(y),
			nm,
			run_fwd: false
		});
		
		self.layer_internals.push(internals);
	}
	
	pub fn new_layer_time_series(&mut self, x_layers: Vec<usize>, internals: InternalTypesCPU,
			y: RNNDataCPU, nm: String) {
		self.layers.push(LayerCPU {
			x_layers,
			y: OutputCPU::RNNData(y),
			nm,
			run_fwd: false
		});
		
		self.layer_internals.push(internals);
	}
}

impl LayerCPU {
	// updates descriptor for changes in sequence lengths for each batch
	pub fn set_output_seq(&mut self, vals: Vec<f32>, seq_len_array: &Vec<c_int>) {
		let max_seq_len = *seq_len_array.iter().max().unwrap();
		debug_assert!((vals.len() % max_seq_len as usize) == 0);
		
		match &mut self.y {
			OutputCPU::RNNData(y) => {
				debug_assert!((max_seq_len as usize *
							seq_len_array.len() *
							y.vec_sz as usize) == vals.len());
				
				y.update_valid_tpoints(seq_len_array, max_seq_len);
				y.mem = vals;
			} OutputCPU::Tensor(y) => {
				let actual_batch_sz = seq_len_array.len() as c_int;
				let batch_sz = max_seq_len * actual_batch_sz;
				// ^ batch tensor dim wraps time (max_seq_len) and actual batch size
				y.update_batch_sz(batch_sz);
				y.mem = vals;
			}
		}
	}
}

pub trait RunCPU {
	fn forward(&mut self, layer_ind: usize, layers: &mut Vec<LayerCPU>);
	
	// default empty:
	fn zero_out_internal_states(&mut self) {}
	fn ld_weights(&mut self, _save_dir: &str, _file_nm: &str) {}
}

