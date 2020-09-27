use std::os::raw::{c_int, c_float};

pub const LAYER_CONFIG_NM: &str = "layers.txt";

macro_rules! impl_output_common{($tensor: ident, $rnn_data: ident) => {
	pub fn tensor(&self) -> &$tensor {
		if let Self::Tensor(tensor) = &self {
			tensor
		}else{
			panic!("could not return tensor");
		}
	}
	
	pub fn tensor_mut(&mut self) -> &mut $tensor {
		if let Self::Tensor(tensor) = self {
			tensor
		}else{
			panic!("could not return tensor");
		}
	}
	
	pub fn rnn_data(&self) -> &$rnn_data {
		if let Self::RNNData(rnn_data) = &self {
			rnn_data
		}else{
			panic!("could not return rnn data");
		}
	}
	
	pub fn rnn_data_mut(&mut self) -> &mut $rnn_data {
		if let Self::RNNData(rnn_data) = self {
			rnn_data
		}else{
			panic!("could not return rnn data");
		}
	}
	
	pub fn ravel_time_shape(&self) -> TensorShape {
		match self {
			Self::Tensor(tensor) => {tensor.shape}
			Self::RNNData(rnn_data) => {rnn_data.ravel_time_shape}
		}
	}
}}

// check that the layer has the correct # of inputs
pub fn check_layer_input_dims(x_layers: &Vec<usize>, layer_type: &String) {
	match layer_type.as_str() {
		//////////////////// 2 inputs
		"add" | "mul" | "MulQK" | "MulSoftmaxQKAndV" | "QKPlusQPosMaskFutureTimesSoftmaxW" => {
			assert!(x_layers.len() == 2, "{}", layer_type);
		
		/////////////////// 1 input
		} "conv" | "pooling" | "softmax" | "softmax_log" | "relu" | "Scale" | "MaskFutureTimes" |
		  "TransposeReshape" | "softmax_across_w" | "MulQAndPos" | "BiasChannels" | "elementwise_affine" |
		  "sum_reduce" | "pow" | "LSTM" | "FullyConnected" | "FullyConnectedWBias" | "Bias" | "QKV" => {
			  assert!(x_layers.len() == 1, "{}", layer_type);
			  
		/////////////////// no input
		} "imgs" | "time_series" => {
			assert!(x_layers.len() == 0, "{}", layer_type);
			
			
		} _ => {
			panic!("unknown layer type: {}", layer_type);
		}
	}
}

#[cfg(not(feature="cpu_only"))]
#[macro_use]
pub mod gpu;
#[cfg(not(feature="cpu_only"))]
pub use gpu::*;

#[macro_use]
pub mod cpu; pub use cpu::*;

/////////////////////////////////////
// layer params for saving/loading

pub struct ImgParams {
	pub shape: TensorShape,
	pub data_type: cudnnDataType_t
}

pub struct ElementwiseAffineParams {
	pub data_type: cudnnDataType_t,
	pub dims: Vec<usize>
}

pub struct TimeSeriesParams {
	pub max_seq_len: c_int,
	pub batch_sz: c_int,
	pub vec_sz: c_int,
	pub data_type: cudnnDataType_t
}

pub struct ActivationParams {
	pub data_type: cudnnDataType_t
}

pub struct SoftmaxParams {
	pub data_type: cudnnDataType_t
}

pub struct BiasParams {
	pub norm_scale: f32,
	pub data_type: cudnnDataType_t
}

pub struct MaxPoolParams {
	pub pool_sz: c_int,
	pub pad_h: c_int,
	pub pad_w: c_int,
	pub stride: c_int,
	pub data_type: cudnnDataType_t
}

pub struct ScaleParams {
	pub alpha: f32,
	pub data_type: cudnnDataType_t
}

pub struct ConvParams {
	pub n_filters: c_int,
	pub filter_sz_h: c_int,
	pub filter_sz_w: c_int,
	pub pad_h: c_int,
	pub pad_w: c_int,
	pub stride: c_int,
	pub norm_scale: f32,
	pub data_type: cudnnDataType_t
}

impl ConvParams {
	pub fn out_shape(&self, x_shape: &TensorShape) -> TensorShape {
		// see documentation of cudnnGetConvolution2dForwardOutputDim() (3.89, pg. 184)
		TensorShape {
			n: x_shape.n,
			c: self.n_filters,
			h: 1 + ( x_shape.h + 2*self.pad_h - ((self.filter_sz_h-1)+1) )/self.stride,
			w: 1 + ( x_shape.w + 2*self.pad_w - ((self.filter_sz_w-1)+1) )/self.stride
		}
	}
}

pub struct LSTMParams {
	pub hidden_sz: c_int,
	pub n_layers: c_int,
	pub dropout_chance: c_float,
	pub norm_scale: f32,
	pub data_type: cudnnDataType_t
}

pub struct MulParams {
	pub data_type: cudnnDataType_t
}

pub struct TransposeReshapeParams {
	pub fwd_dims: Vec<usize>, // ex [1,2,0,3], which is similar to running np.transpose()
	pub new_shape: TensorShape,
	pub data_type: cudnnDataType_t
}

pub struct FullyConnectedParams {
	pub vec_out_sz: i32, // output dimension size
	pub norm_scale: f32,
	pub data_type: cudnnDataType_t
}

pub struct FullyConnectedWBiasParams {
	pub vec_out_sz: i32, // output dimension size
	pub weight_initialization: WeightInitialization,
	pub bias_initialization: WeightInitialization,
	//pub relu: bool, // when true, relu applied: relu(w*x + b)
	pub data_type: cudnnDataType_t
}

//////////// multi head layer components

#[derive(Copy, Clone)]
pub struct MultiHeadAttnParams {
	pub n_heads: c_int,
	pub feed_forward_sz: c_int,
	pub data_type: cudnnDataType_t
}

pub struct QKVLayerParams {
	pub weight_initialization: WeightInitialization,
	pub bias_initialization: WeightInitialization,
	pub n_heads: c_int,
	pub data_type: cudnnDataType_t
}

pub struct MulQKParams {
	pub data_type: cudnnDataType_t
}

pub struct MulQAndPosParams {
	pub norm_scale: f32,
	pub data_type: cudnnDataType_t
}

pub struct MulSoftmaxQKAndVParams {
	pub data_type: cudnnDataType_t
}

pub struct MaskFutureTimesParams {
	pub scale: f32,
	pub data_type: cudnnDataType_t
}

pub struct QKPlusQPosMaskFutureTimesSoftmaxWParams {
	pub weight_initialization: WeightInitialization, // for the relative positions
	pub scale: f32, // applied after Q*K & Q*pos are added together
	pub data_type: cudnnDataType_t
}

/////////////////////////////////////
//////// sum
#[derive(Clone)]
pub enum SumType {
	All,
	Axes(Vec<usize>)
}

#[derive(Clone)]
pub struct SumReduceParams {
	pub sum_type: SumType,
	pub data_type: cudnnDataType_t
}

use std::fmt;
impl fmt::Display for SumType {
	fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
		write!(f, "{}", match self {
				SumType::All => {String::from("All")}
				SumType::Axes(axes) => {
					let mut txt = String::new();
					for (i, ax) in axes.iter().enumerate() {
						txt.push_str(&format!("{}", ax));
						if i != (axes.len()-1) {
							txt.push(',');
						}
					}
					txt
				}
		})
	}
}

use std::num::ParseIntError;
use std::str::FromStr;
impl FromStr for SumType {
	type Err = ParseIntError;
	fn from_str(s: &str) -> Result<Self, Self::Err> {
		Ok(match s {
			"All" => {SumType::All}
			_ => {
				let txt_strs: Vec<&str> = s.split(",").collect();
				let mut axes = Vec::with_capacity(txt_strs.len());
				for txt_str in txt_strs {
					axes.push(if let Result::Ok(val) = txt_str.trim().to_string().parse() {
							val
						}else{panic!("Cannot parse: \"{}\"", s);}
					);
				}
				SumType::Axes(axes)
			}
		})
	}
}

