use std::fmt;
use super::*;
use std::num::ParseIntError;
use std::str::FromStr;

#[cfg(not(feature="cpu_only"))]
impl fmt::Display for cudnnConvolutionFwdAlgo_t {
	fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
		write!(f, "{}", match self {
			cudnnConvolutionFwdAlgo_t::CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM => {"CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEM"}
			cudnnConvolutionFwdAlgo_t::CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM => {"CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEM"}
			cudnnConvolutionFwdAlgo_t::CUDNN_CONVOLUTION_FWD_ALGO_GEMM => {"CUDNN_CONVOLUTION_FWD_ALGO_GEM"}
			cudnnConvolutionFwdAlgo_t::CUDNN_CONVOLUTION_FWD_ALGO_DIRECT => {"CUDNN_CONVOLUTION_FWD_ALGO_DIRECT"}
			cudnnConvolutionFwdAlgo_t::CUDNN_CONVOLUTION_FWD_ALGO_FFT => {"CUDNN_CONVOLUTION_FWD_FFT"}
			cudnnConvolutionFwdAlgo_t::CUDNN_CONVOLUTION_FWD_ALGO_FFT_TILING => {"CUDNN_CONVOLUTION_FWD_FFT_TILING"}
			cudnnConvolutionFwdAlgo_t::CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD => {"CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD"}
			cudnnConvolutionFwdAlgo_t::CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD_NONFUSED => {"CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD_NONFUSED"}
			cudnnConvolutionFwdAlgo_t::CUDNN_CONVOLUTION_FWD_ALGO_COUNT => {"CUDNN_CONVOLUTION_FWD_ALGO_COUNT"}
		})
	}
}

impl fmt::Display for cudnnDataType_t {
	fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
		write!(f, "{}", match self {
				cudnnDataType_t::CUDNN_DATA_FLOAT => {"float"}
				cudnnDataType_t::CUDNN_DATA_DOUBLE => {"double"}
				cudnnDataType_t::CUDNN_DATA_HALF => {"half"}
				cudnnDataType_t::CUDNN_DATA_INT8 => {"int8"}
				cudnnDataType_t::CUDNN_DATA_INT32 => {"int32"}
				cudnnDataType_t::CUDNN_DATA_INT8x4 => {"int8x4"}
				cudnnDataType_t::CUDNN_DATA_UINT8 => {"uint8"}
				cudnnDataType_t::CUDNN_DATA_UINT8x4 => {"uint8x4"}
				cudnnDataType_t::CUDNN_DATA_INT8x32 => {"int8x32"}
		})
	}
}

impl FromStr for cudnnDataType_t {
	type Err = ParseIntError;
	fn from_str(s: &str) -> Result<Self, Self::Err> {
		Ok(match s {
			"float" => {cudnnDataType_t::CUDNN_DATA_FLOAT}
			"double" => {cudnnDataType_t::CUDNN_DATA_DOUBLE}
			"half" => {cudnnDataType_t::CUDNN_DATA_HALF}
			"int8" => {cudnnDataType_t::CUDNN_DATA_INT8}
			"int32" => {cudnnDataType_t::CUDNN_DATA_INT32}
			"int8x4" => {cudnnDataType_t::CUDNN_DATA_INT8x4}
			"uint8" => {cudnnDataType_t::CUDNN_DATA_UINT8}
			"uint8x4" => {cudnnDataType_t::CUDNN_DATA_UINT8x4}
			"int8x32" => {cudnnDataType_t::CUDNN_DATA_INT8x32}
			_ => {panic!("unknown data type: {}", s)}
		})
	}
}

pub fn print_vec<T: fmt::Display>(vals: &Vec<T>, config_txt: &mut String) {
	if vals.len() == 0 {return;}
	
	for (i, x_ind) in vals.iter().enumerate() {
		config_txt.push_str(&format!("{}", x_ind));
		if i != (vals.len() - 1) {
			config_txt.push_str(", ");
		}
	}
	config_txt.push('\n');
}

////////////
impl fmt::Display for WeightInitialization {
	fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
		write!(f, "{}", match self {
			Self::NormScale(norm_scale) => {
				format!("NormScale ({})", norm_scale)
			} Self::XavierUniform(fan_out, fan_in) => {
				format!("XavierUniform ({}, {})", fan_out, fan_in)
			}
		})
	}
}

impl FromStr for WeightInitialization {
	type Err = ParseIntError;
	fn from_str(s: &str) -> Result<Self, Self::Err> {
		let opening_paren_split = s.split("(").collect::<Vec<&str>>();
		assert!(opening_paren_split.len() == 2);
		
		let weight_type = opening_paren_split[0].trim();
		
		// remove parentheses
		let vals_str: &str = opening_paren_split[1].split(")").collect::<Vec<&str>>()[0];
		
		match weight_type {
			"NormScale" =>  {
				if let Ok(norm_scale) = vals_str.trim().to_string().parse() {
					Ok(Self::NormScale(norm_scale))
				}else{
					panic!("failed parsing norm scale weight initialization parameter");
				}
			}
			"XavierUniform" => {
				let dims = vals_str.split(",").collect::<Vec<&str>>();
				assert!(dims.len() == 2);
				Ok(Self::XavierUniform(
						dims[0].trim().to_string().parse()?,
						dims[1].trim().to_string().parse()?
				))
			}
			_ => {panic!("unknown weight initialization type");}
		}
	}
}

//////////// shapes
impl ToString for TensorShape {
	fn to_string(&self) -> String {
		format!("({}, {}, {}, {})", self.n, self.c, self.h, self.w)
	}
}

impl ToString for FilterShape {
	fn to_string(&self) -> String {
		format!("({}, {}, {}, {})", self.k, self.c, self.h, self.w)
	}
}

impl ToString for Filter3Shape {
	fn to_string(&self) -> String {
		format!("({}, {}, {})", self.dim1, self.dim2, self.dim3)
	}
}

impl FromStr for TensorShape {
	type Err = ParseIntError;
	fn from_str(s: &str) -> Result<Self, Self::Err> {
		// remove parentheses
		let vals_str: &str = s.split("(").collect::<Vec<&str>>()[1]
					.split(")").collect::<Vec<&str>>()[0];
			
		let vals_split: Vec<&str> = vals_str.split(",").collect();
		assert!(vals_split.len() == 4);
		
		Ok(Self {n: vals_split[0].trim().to_string().parse()?,
				 c: vals_split[1].trim().to_string().parse()?,
				 h: vals_split[2].trim().to_string().parse()?,
				 w: vals_split[3].trim().to_string().parse()?,
		})
	}
}

impl FromStr for Filter3Shape {
	type Err = ParseIntError;
	fn from_str(s: &str) -> Result<Self, Self::Err> {
		// remove parentheses
		let vals_str: &str = s.split("(").collect::<Vec<&str>>()[1]
					.split(")").collect::<Vec<&str>>()[0];
	
		let vals_split: Vec<&str> = vals_str.split(",").collect();
		assert!(vals_split.len() == 3);
		
		Ok(Self {dim1: vals_split[0].trim().to_string().parse()?,
				 dim2: vals_split[1].trim().to_string().parse()?,
				 dim3: vals_split[2].trim().to_string().parse()?,
		})
	}
}

pub fn num_format<T: std::fmt::Display>(n: T) -> String {
	let mut n_str = format!("{}", n);
	
	// number needs commas inserted
	if n_str.len() > 3 {
		// start from back and work to higher decimal places
		let mut insert_pos = n_str.len() - 3;
		loop {
			n_str.insert(insert_pos, ',');
			if insert_pos <= 3 {break;}
			insert_pos -= 3;
		}
	}

	n_str
}

