#![allow(non_camel_case_types)]
use std::os::raw::{c_int};

#[cfg(not(feature="cpu_only"))]
use super::f16;

use std::cmp::max;
#[cfg(not(feature="cpu_only"))]
use crate::cudnn::*;
pub use crate::cudnn_common::*;

#[cfg(not(feature="cpu_only"))]
pub mod gpu;
#[cfg(not(feature="cpu_only"))]
pub use gpu::*;

pub mod cpu; pub use cpu::*;

pub type size_t = usize;

pub const TENSOR_FORMAT: cudnnTensorFormat_t = cudnnTensorFormat_t::CUDNN_TENSOR_NCHW;
pub const RNN_DATA_LAYOUT: cudnnRNNDataLayout_t = cudnnRNNDataLayout_t::CUDNN_RNN_DATA_LAYOUT_SEQ_MAJOR_UNPACKED;

#[derive(Copy, Clone)]
pub enum WeightInitialization {
	NormScale(f32),
	XavierUniform(i32, i32) // fan_out, fan_in
}

//pub struct f16(u16);

#[derive(Copy, Clone, PartialEq)]
pub struct TensorShape {
	pub n: c_int, // imgs
	pub c: c_int, // channels
	pub h: c_int,
	pub w: c_int
}

pub struct Tensor3Shape {
	pub dim1: c_int,
	pub dim2: c_int,
	pub dim3: c_int
}

#[derive(Copy, Clone, PartialEq)]
pub struct FilterShape {
	pub k: c_int, // number of filters
	pub c: c_int, // number of input channels
	pub h: c_int,
	pub w: c_int
}

pub struct Filter3Shape {
	pub dim1: c_int,
	pub dim2: c_int,
	pub dim3: c_int
}

impl From<&FilterShape> for TensorShape {
	fn from(filter_shape: &FilterShape) -> Self {
		Self {
			n: filter_shape.k,
			c: filter_shape.c,
			h: filter_shape.h,
			w: filter_shape.w
		}
	}
}

#[cfg(not(feature="cpu_only"))]
impl From<&RNNData> for TensorShape {
	fn from(rnn_data: &RNNData) -> Self {
		Self {
			n: rnn_data.max_seq_len,
			c: rnn_data.batch_sz,
			h: rnn_data.vec_sz,
			w: 1
		}
	}
}

impl TensorShape {
	pub fn n_elements(&self) -> size_t {
		self.n as size_t *
		self.c as size_t *
		self.h as size_t *
		self.w as size_t
	}
	
	pub fn broadcast(shape1: TensorShape, shape2: TensorShape) -> Self {
		debug_assert!(shape1.n == shape2.n || shape1.n == 1 || shape2.n == 1);
		debug_assert!(shape1.c == shape2.c || shape1.c == 1 || shape2.c == 1);
		debug_assert!(shape1.h == shape2.h || shape1.h == 1 || shape2.h == 1);
		debug_assert!(shape1.w == shape2.w || shape1.w == 1 || shape2.w == 1);
		
		Self {
				n: max(shape1.n, shape2.n),
				c: max(shape1.c, shape2.c),
				h: max(shape1.h, shape2.h),
				w: max(shape1.w, shape2.w)
		}
	}
	
	pub fn print(&self) {
		println!("tensor shape: ({}, {}, {}, {})", self.n, self.c, self.h, self.w);
	}
}

impl Tensor3Shape {
	pub fn n_elements(&self) -> size_t {
		self.dim1 as size_t *
		self.dim2 as size_t *
		self.dim3 as size_t
	}
}

impl FilterShape {
	pub fn n_elements(&self) -> size_t {
		self.k as size_t *
		self.c as size_t *
		self.h as size_t *
		self.w as size_t
	}
	
	pub fn print(&self) {
		println!("filter shape: ({}, {}, {}, {})", self.k, self.c, self.h, self.w);
	}
}

impl Filter3Shape {
	pub fn n_elements(&self) -> size_t {
		self.dim1 as size_t *
		self.dim2 as size_t *
		self.dim3 as size_t
	}
}

