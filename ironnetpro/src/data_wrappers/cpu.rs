use std::os::raw::{c_int, c_float};
use super::*;
use crate::crossbeam::scope;
use crate::rand::XorState;
#[cfg(feature="profile")]
use crate::profiling_internal::*;

#[derive(Clone)]
pub struct TensorCPU {
	pub mem: Vec<c_float>,
	pub shape: TensorShape
}

#[derive(Clone)]
pub struct RNNDataCPU {
	pub mem: Vec<c_float>, // [max_seq_len, batch_sz, vec_sz]
	
	pub ravel_time_shape: TensorShape, // timesteps raveled into batch dim
	
	pub max_seq_len: c_int,
	pub batch_sz: c_int,
	pub vec_sz: c_int,
	pub seq_len_array: Vec<c_int>
}

pub struct FilterCPU {
	pub mem: Vec<c_float>,
	pub shape: FilterShape
}

pub struct Filter3CPU {
	pub mem: Vec<c_float>,
	pub shape: FilterShape
}

impl TensorCPU {	
	pub fn update_batch_sz(&mut self, batch_sz: c_int) {
		if self.shape.n == batch_sz || self.mem.len() == 1 {return;} // nothing to change
		self.shape.n = batch_sz;
		debug_assert!(self.mem.len() >= self.shape.n_elements(),
				"attempted to increase tensor batch_size beyond memory allocation {} -> {} batch_sz: {}",
				self.mem.len(), self.shape.n_elements(), batch_sz);
	}
	
	pub fn new3(shape3: &Tensor3Shape) -> Self {
		Self {
			mem: vec!{0.; shape3.n_elements()},
			shape: TensorShape {n: shape3.dim1,
						  c: shape3.dim2,
						  h: shape3.dim3,
						  w: 1}
		}
	}
	
	/*pub fn new(shape: &TensorShape) -> Self {
		Self {
			mem: vec!{0.; shape.n_elements()},
			shape
		}
	}*/

	pub fn zeros3(shape: &Tensor3Shape) -> Self {Self::new3(shape)}
}

impl RNNDataCPU {
	pub fn new(max_seq_len: c_int, batch_sz: c_int, vec_sz: c_int, seq_len_array: &Vec<c_int>) -> Self {
		debug_assert!(RNN_DATA_LAYOUT == cudnnRNNDataLayout_t::CUDNN_RNN_DATA_LAYOUT_SEQ_MAJOR_UNPACKED);
		let ravel_time_shape = TensorShape {
					n: max_seq_len * batch_sz,
					c: 1,
					h: vec_sz, 
					w: 1
		};
		
		Self {
			mem: vec!{0.; (max_seq_len*batch_sz*vec_sz) as usize},
			
			ravel_time_shape, // used in layers which don't deal with time (time is raveled into the batch dimension)
			
			//////
			max_seq_len,
			batch_sz,
			vec_sz,
			seq_len_array: seq_len_array.clone()
		}
	}
	
	// each batch example may have a sequence different than all other batch examples,
	// but it should not exceed max_seq_len
	pub fn update_valid_tpoints(&mut self, seq_len_array: &Vec<c_int>, max_seq_len: c_int) {
		debug_assert!(self.mem.len() >= (max_seq_len*self.batch_sz*self.vec_sz) as usize,
				"insufficient memory allocated for sequence {} -> {}, max_seq_len {} batch_sz {} vec_sz {}",
				self.mem.len(), max_seq_len*self.batch_sz*self.vec_sz, max_seq_len, self.batch_sz, self.vec_sz);
		
		self.max_seq_len = max_seq_len;
		self.seq_len_array = seq_len_array.clone();
		
		// for layers which do not deal with time, ravel the batch dimension to include the time dimension too
		self.ravel_time_shape.n = max_seq_len * self.batch_sz;
	}
	
	pub fn zeros(max_seq_len: c_int, batch_sz: c_int, vec_sz: c_int, seq_len_array: &Vec<c_int>) -> Self {
		Self::new(max_seq_len, batch_sz, vec_sz, seq_len_array)
	}
	
	pub fn zero_out(&mut self) {
		for v in self.mem.iter_mut() {
			*v = 0.;
		}
	}
}

macro_rules! impl_filter{($filter: ty, $shape: ty) => {
	impl $filter {
		pub fn new(shape: $shape) -> Self {
			debug_assert!(TENSOR_FORMAT == cudnnTensorFormat_t::CUDNN_TENSOR_NCHW);
			Self {
				mem: vec!{0.; shape.n_elements()},
				shape
			}
		}
		
		pub fn zeros(shape: $shape) -> Self {Self::new(shape)}
		
		pub fn new_norm(shape: $shape, norm_scale: f32, rng: &mut XorState) -> Self {
			let mut filter = Self::new(shape);
			filter.mem = rng.gen_norm_vec(shape.n_elements(), norm_scale);
			
			filter
		}
		
		pub fn new_uniform(shape: $shape, norm_scale: f32, rng: &mut XorState) -> Self {
			let mut filter = Self::new(shape);
			filter.mem = rng.gen_vec(shape.n_elements(), norm_scale);
			
			filter
		}
		
		pub fn zero_out(&mut self) {
			for v in self.mem.iter_mut() {
				*v = 0.;
			}
		}
	}
}}

impl_filter!(TensorCPU, TensorShape);
impl_filter!(FilterCPU, FilterShape);
impl_filter!(Filter3CPU, FilterShape);


use std::ops::{Add, Mul};
////////////////////////////////////////////////////////////////////
// add

// for equal shaped tensors
impl Add for &TensorCPU {
	type Output = TensorCPU;
	
	fn add(self, b: Self) -> TensorCPU {
		let shape = TensorShape::broadcast(self.shape, b.shape);
		let mut mem = Vec::with_capacity(shape.n_elements());
		debug_assert!(self.shape == b.shape);
		
		for (av, bv) in self.mem.iter().take(self.shape.n_elements())
				.zip(b.mem.iter()) {
			mem.push(*av + *bv);
		}
		
		TensorCPU {
			mem,
			shape
		}
	}
}

macro_rules! rnndata_plus_rnndata{($type: ty) => {
	impl Add for $type {
		type Output = RNNDataCPU;
		
		fn add(self, b: Self) -> RNNDataCPU {
			#[cfg(feature="profile")]
			let _g = Guard::new("add RNNDataCPU + RNNDataCPU");
			
			let shape = self.ravel_time_shape;
			debug_assert!(shape == b.ravel_time_shape);
			let mut y = RNNDataCPU::new(self.max_seq_len, self.batch_sz, self.vec_sz, &self.seq_len_array);
			
			for ((av, bv), yv) in self.mem.iter().take(shape.n_elements())
					.zip(b.mem.iter()).zip(y.mem.iter_mut()) {
				*yv = *av + *bv;
			}
			
			y
		}
	}
}}

rnndata_plus_rnndata!(RNNDataCPU);
rnndata_plus_rnndata!(&RNNDataCPU);

// RNNDataCPU: [n_imgs, 1, 1, k]
// Filter3CPU: [1, k, 1, 1]

macro_rules! rnndata_plus_filter3{($type: ty) => {
	impl Add<&Filter3CPU> for $type {
		type Output = RNNDataCPU;
		fn add(self, b: &Filter3CPU) -> RNNDataCPU {
			#[cfg(feature="profile")]
			let _g = Guard::new("add RNNDataCPU + Filter3CPU");
			
			let mut y = RNNDataCPU::new(self.max_seq_len, self.batch_sz, self.vec_sz, &self.seq_len_array);
			
			let n_imgs = self.ravel_time_shape.n as usize;
			let k = self.ravel_time_shape.n_elements() / n_imgs;
			
			debug_assert!(b.shape.n_elements() == k);
			
			for img in 0..n_imgs {
				for ((yv, sv), bv) in y.mem.iter_mut().skip(img*k).take(k)
						     .zip(self.mem.iter().skip(img*k).take(k))
						     .zip(b.mem.iter()) {
					*yv = *sv + *bv;
				}
			}
			
			y
		}
	}
}}

rnndata_plus_filter3!(RNNDataCPU);
rnndata_plus_filter3!(&RNNDataCPU);

/////////////////////////////////////////////////////////////////////////
// mul

// point-wise
impl Mul for &TensorCPU {
	type Output = TensorCPU;
	
	fn mul(self, b: Self) -> TensorCPU {
		#[cfg(feature="profile")]
		let _g = Guard::new("point-wise mul TensorCPU .* TensorCPU");
		
		let shape = TensorShape::broadcast(self.shape, b.shape);
		let mut mem = Vec::with_capacity(shape.n_elements());
		debug_assert!(self.shape == b.shape);
		
		for (av, bv) in self.mem.iter().take(self.shape.n_elements())
				.zip(b.mem.iter()) {
			mem.push(*av * *bv);
		}
		
		TensorCPU {
			mem,
			shape
		}
	}
}

// point-wise

macro_rules! rnndata_pointwise_mul{($type: ty) => {
	impl Mul<$type> for RNNDataCPU {
		type Output = RNNDataCPU;
		
		fn mul(self, b: $type) -> RNNDataCPU {
			#[cfg(feature="profile")]
			let _g = Guard::new("point-wise mul RNNDataCPU .* [TensorCPU | RNNDataCPU]");
			
			let n_elements = self.ravel_time_shape.n_elements();
			
			let mut y = RNNDataCPU::new(self.max_seq_len, self.batch_sz, self.vec_sz, &self.seq_len_array);
			
			for ((yv, sv), bv) in y.mem.iter_mut().take(n_elements)
					.zip(self.mem.iter())
					.zip(b.mem.iter()) {
				*yv = (sv * bv);
			}
			
			y
		}
	}
}}

rnndata_pointwise_mul!(TensorCPU);
rnndata_pointwise_mul!(&TensorCPU);
rnndata_pointwise_mul!(RNNDataCPU);

////////////////////////////////////////////////////////////////////////
// matrix mul

// filter3CPU.mat_mul_vec():
pub trait MatMulVec<T> {
	fn mat_mul_rnn(&self, b: T) -> RNNDataCPU;
}

// Filter3CPU: [1, out_sz, vec_sz, 1]
// RNNDataCPU: [batch_sz, 1, vec_sz, 1]
//
// output: [batch_sz, 1, out_sz, 1]
macro_rules! filter3_matmul_rnndata{($type: ty) => {
	impl MatMulVec<$type> for &Filter3CPU {
		fn mat_mul_rnn(&self, b: $type)-> RNNDataCPU {
			#[cfg(feature="profile")]
			let _g = Guard::new("mat_mul_rnn &Filter3CPU * &RNNDataCPU");
			
			// (filter and tensor shape fields: [k,c,h,w], [n,c,h,w])
			debug_assert!(self.shape.k == 1);
			debug_assert!(self.shape.w == 1);
			debug_assert!(self.shape.w == b.ravel_time_shape.w);
			
			let n_imgs = b.ravel_time_shape.n as usize;
			let out_sz = self.shape.c as usize;
			let vec_sz = self.shape.h as usize;
			
			let mut y = RNNDataCPU::new(b.max_seq_len, b.batch_sz, out_sz as c_int, &b.seq_len_array);
			debug_assert!(y.ravel_time_shape == TensorShape {
						n: n_imgs as c_int,
						c: 1,
						h: out_sz as c_int,
						w: 1
			});
			
			matmul_rnndata_internals!(out_sz, vec_sz, y, self, b);
			
			y
		}
	}
}}

// Filter3CPU: [1, out_sz, vec_sz, 1] `A`
// TensorCPU: [batch_sz, vec_sz, 1, 1] `b`
//
// output: [batch_sz, 1, out_sz, 1] `y`
macro_rules! matmul_rnndata_internals{($out_sz: expr, $vec_sz: expr, $y: expr, $A: expr, $b: expr) => {
	// idea to use crossbeam scope:
	// https://stackoverflow.com/questions/33818141/how-do-i-pass-disjoint-slices-from-a-vector-to-different-threads (Accessed April 8, 2020)
	const N_THREADS: usize = 4;
	let vars_per_thread = $out_sz / N_THREADS;
	
	// compute y values for each image, multiplying self.mem with the bias [vec_sz] for each img
	for (yimg, bimg) in $y.mem.chunks_mut($out_sz).zip(
			   	  $b.mem.chunks($vec_sz)) {
		// yimg: [out_sz]
		// bimg: [vec_sz]
		scope(|s| {
			// for each thread iterate over `vars_per_thread` (chunk up `out_sz` and split across threads)
			for (thread_i, yimg_thread) in yimg.chunks_mut(vars_per_thread).enumerate() {
				s.spawn(move |_| {
					// y[i] = (self.mem[i,:vec_sz] * b[:vec_sz]).sum()
					for (out_i, yv) in yimg_thread.iter_mut().enumerate() {
						*yv = $A.mem.iter()
							.skip((out_i + thread_i*vars_per_thread)*$vec_sz) // starting index of the y value we're computing
							.take($vec_sz)
							.zip(bimg)
							.fold(0., |acc, (sv, bv)| acc + *sv * *bv);
					}
				});
			}
			
			// the following is no faster than the above (and is more confusing, but looks like it'd be faster)
			/*for ((thread_i, yimg_thread), sthread) in yimg.chunks_mut(vars_per_thread).enumerate()
								.zip(self.mem.chunks(vars_per_thread*vec_sz)) {
				s.spawn(move |_| {
					for ((out_i, yv), sthreadv) in yimg_thread.iter_mut().enumerate()
								.zip(sthread.chunks(vec_sz)) {
						*yv = sthreadv.iter()
							.zip(bimg)
							.fold(0., |acc, (sv, bv)| acc + *sv * *bv);
					}
				});
			}*/
		}).unwrap();
	}
}}

filter3_matmul_rnndata!(RNNDataCPU);
filter3_matmul_rnndata!(&RNNDataCPU);

const MAX_SEQ_LEN_FIXED: c_int = 1;

// Filter3CPU: [1, out_sz, vec_sz, 1] `self`
// TensorCPU: [batch_sz, vec_sz, 1, 1] `b`
//
// output: [batch_sz, 1, out_sz, 1]
impl MatMulVec<&TensorCPU> for &Filter3CPU {
	fn mat_mul_rnn(&self, b: &TensorCPU)-> RNNDataCPU {
		#[cfg(feature="profile")]
		let _g = Guard::new("mat_mul_rnn &Filter3CPU * &RNNDataCPU");
		
		// (filter and tensor shape fields: [k,c,h,w], [n,c,h,w])
		debug_assert!(self.shape.k == 1);
		debug_assert!(self.shape.w == 1);
		debug_assert!(self.shape.w == b.shape.w);
		
		let n_imgs = b.shape.n as usize;
		let out_sz = self.shape.c as usize;
		let vec_sz = self.shape.h as usize;
		
		let mut y = RNNDataCPU::new(MAX_SEQ_LEN_FIXED, 
				n_imgs as c_int, out_sz as c_int, &vec!{1; n_imgs});
		
		debug_assert!(y.ravel_time_shape == TensorShape {
					n: n_imgs as c_int,
					c: 1,
					h: out_sz as c_int,
					w: 1
		});
		
		matmul_rnndata_internals!(out_sz, vec_sz, y, self, b);
			
		y
	}
}

////////////////////////////////////////////////////////////////////////
// tanh & sigmoid
impl RNNDataCPU {
	pub fn tanh(&self) -> Self {
		#[cfg(feature="profile")]
		let _g = Guard::new("sigmoid RNNDataCPU");
		
		let n_elements = self.ravel_time_shape.n_elements();
		
		let mut y = RNNDataCPU::new(self.max_seq_len, self.batch_sz, self.vec_sz, &self.seq_len_array);
		for (yv, v) in y.mem.iter_mut().take(n_elements)
				.zip(self.mem.iter().take(n_elements)) {
			*yv = v.tanh();
		}
		y
	}
	
	pub fn sigmoid(&self) -> Self {
		#[cfg(feature="profile")]
		let _g = Guard::new("sigmoid RNNDataCPU");
		
		let n_elements = self.ravel_time_shape.n_elements();
		
		let mut y = RNNDataCPU::new(self.max_seq_len, self.batch_sz, self.vec_sz, &self.seq_len_array);
		for (yv, v) in y.mem.iter_mut().take(n_elements)
				.zip(self.mem.iter().take(n_elements)) {
			*yv = 1. / (1. + (-v).exp());
		}
		y
	}
}

impl TensorCPU {
	pub fn tanh(&self) -> Self {
		#[cfg(feature="profile")]
		let _g = Guard::new("tanh TensorCPU");

		let n_elements = self.shape.n_elements();
		let mut y = TensorCPU::new(self.shape);
		for (yv, v) in y.mem.iter_mut().take(n_elements)
				.zip(self.mem.iter().take(n_elements)) {
			*yv = v.tanh();
		}
		y
	}
}

///////////////////////////////////////////////////////////////////////
// conversions
impl From<TensorCPU> for RNNDataCPU {
	fn from(tensor: TensorCPU) -> Self {
		let batch_sz = tensor.shape.n;
		let vec_sz = tensor.shape.n_elements() as c_int / batch_sz;
		
		let mut y = RNNDataCPU::new(MAX_SEQ_LEN_FIXED, batch_sz, vec_sz,
				&vec!{1; batch_sz as usize});
		
		y.mem = tensor.mem;
		
		y
	}
}

impl From<RNNDataCPU> for TensorCPU {
	fn from(rnndata: RNNDataCPU) -> Self {
		let n_imgs = rnndata.batch_sz;
		let vec_sz = rnndata.vec_sz;
		
		debug_assert!(rnndata.max_seq_len == 1);
		
		let mut y = TensorCPU::new3(&Tensor3Shape {
			dim1: n_imgs,
			dim2: vec_sz,
			dim3: 1
		});
		y.mem = rnndata.mem;
		
		y
	}
}

