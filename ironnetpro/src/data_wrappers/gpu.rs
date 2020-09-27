use std::os::raw::{c_int};
use crate::cudnn::*;
use crate::rand::XorState;
use super::*;

pub struct Tensor {
	pub desc: TensorDescriptor,
	pub mem: gpuMem,
	pub shape: TensorShape
}

pub struct Filter {
	pub desc: FilterDescriptor,
	pub tensor_desc: TensorDescriptor,
	pub mem: gpuMem,
	pub shape: FilterShape
}

pub struct Filter3 {
	pub desc: FilterDescriptor,
	pub tensor_desc: TensorDescriptor,
	pub mem: gpuMem,
	pub shape: FilterShape
}

pub struct RNNData {
	pub mem: gpuMem,
	
	// RNNDataDescriptors (used in layers/lstm.rs)
	pub desc: RNNDataDescriptor, // training descriptor
	pub inference_desc: RNNDataDescriptor, // seq length of 1 timestep
	
	// Tensor descriptors (used in layers/lstm.rs)
	pub timestep_shape: Tensor3Shape, // shape at each timestep
	
	pub timestep_tensor_descs: Vec<TensorDescriptor>, // used in layers/lstm.rs for workspace sizes; represents tensor desc at time [t]
	pub timestep_tensor_desc_vals: Vec<cudnnTensorDescriptor_t>,
	
	// Tensor descriptors (ravels time and batch size together)
	pub ravel_time_shape: TensorShape, // timesteps raveled into batch dim
	pub ravel_time_tensor_desc: TensorDescriptor, // timesteps raveled into batch dim
	
	pub data_type: cudnnDataType_t,
	pub max_seq_len: c_int,
	pub batch_sz: c_int,
	pub vec_sz: c_int,
	pub seq_len_array: Vec<c_int>
}

impl Tensor {
	pub fn new(data_type: cudnnDataType_t, shape: TensorShape) -> Self {
		debug_assert!(TENSOR_FORMAT == cudnnTensorFormat_t::CUDNN_TENSOR_NCHW);
		Self {
			desc: TensorDescriptor::new(data_type, TENSOR_FORMAT, shape),
			mem: gpuMem::new(data_type, shape.n_elements()),
			shape
		}
	}
	
	pub fn new3(data_type: cudnnDataType_t, shape3: &Tensor3Shape) -> Self {
		Self {
			desc: TensorDescriptor::new3(data_type, shape3),
			mem: gpuMem::new(data_type, shape3.n_elements()),
			shape: TensorShape {n: shape3.dim1,
						  c: shape3.dim2,
						  h: shape3.dim3,
						  w: 1}
		}
	}
	
	pub fn zeros(data_type: cudnnDataType_t, shape: TensorShape) -> Self {
		let tensor = Self::new(data_type, shape);
		tensor.zero_out();
		tensor
	}
	
	pub fn zeros3(data_type: cudnnDataType_t, shape: &Tensor3Shape) -> Self {
		debug_assert!(data_type == cudnnDataType_t::CUDNN_DATA_FLOAT);
		
		let tensor = Self::new3(data_type, shape);
		tensor.zero_out();
		tensor
	}
	
	pub fn zero_out(&self) {
		self.mem.zero_out();
	}
	
	pub fn one_out(&self) {
		self.mem.one_out();
	}

	pub fn ret(&self) -> Vec<f32> {
		self.mem.ret(self.shape.n_elements())
	}
	
	pub fn set<T>(&self, src: &Vec<T>) {
		self.mem.set(src);
	}
	
	pub fn update_batch_sz(&mut self, batch_sz: c_int) {
		if self.shape.n == batch_sz || self.mem.n_elements == 1 {return;} // nothing to change
		
		let shape_n_orig = self.shape.n;
		self.shape.n = batch_sz;
		assert!(self.mem.n_elements >= self.shape.n_elements(),
				"attempted to increase tensor batch_size beyond memory allocation {} -> {};  attempted setting batch_sz: {} for current shape.n: {}",
				self.mem.n_elements, self.shape.n_elements(), batch_sz, shape_n_orig);

		self.desc = TensorDescriptor::new(self.mem.dataType, TENSOR_FORMAT, self.shape);
	}
}

impl RNNData {
	pub fn new(data_type: cudnnDataType_t, max_seq_len: c_int, 
			batch_sz: c_int, vec_sz: c_int,
			seq_len_array: &Vec<c_int>) -> Self {
		debug_assert!(RNN_DATA_LAYOUT == cudnnRNNDataLayout_t::CUDNN_RNN_DATA_LAYOUT_SEQ_MAJOR_UNPACKED);
		// ^ if this changes, the Tensor3Shape ordering returned by this fn would need to be changed
		
		let timestep_shape = Tensor3Shape { // shape at each timestep
					dim1: batch_sz,
					dim2: vec_sz,
					dim3: 1
		};
		
		let ravel_time_shape = TensorShape {
					n: max_seq_len * batch_sz,
					c: 1,
					h: vec_sz, 
					w: 1
		};
		
		let ravel_time_tensor_desc = TensorDescriptor::new(data_type, cudnnTensorFormat_t::CUDNN_TENSOR_NCHW, ravel_time_shape);
		
		let mut rnn_data = Self {
			mem: gpuMem::new(data_type, (max_seq_len*batch_sz*vec_sz) as usize),
			
			//// rnnDataDescriptors used in layers/lstm.rs
			desc: RNNDataDescriptor::new(data_type, RNN_DATA_LAYOUT, max_seq_len, batch_sz, vec_sz, seq_len_array),
			inference_desc: RNNDataDescriptor::new(data_type, RNN_DATA_LAYOUT, 1, batch_sz, vec_sz,
											&vec!{1 as c_int; batch_sz as usize}),
			
			//// tensor descriptors
			timestep_shape, // used in layers/lstm.rs for initialization of structures
			timestep_tensor_descs: Vec::new(),
			timestep_tensor_desc_vals: Vec::new(),
			
			ravel_time_shape, // used in layers which don't deal with time (time is raveled into the batch dimension)
			ravel_time_tensor_desc,
			
			//////
			data_type,
			max_seq_len,
			batch_sz,
			vec_sz,
			seq_len_array: seq_len_array.clone()
		};
		
		rnn_data.update_timestep_tensor_descs();
		rnn_data
	}
	
	// batch size at each timestamp
	fn update_timestep_tensor_descs(&mut self) {
		self.timestep_tensor_descs = Vec::with_capacity(self.max_seq_len as usize);
		self.timestep_tensor_desc_vals = Vec::with_capacity(self.max_seq_len as usize);
		
		for t in 1..=self.max_seq_len {
			let n_seqs_included = self.seq_len_array.iter()
				.filter(|&&v| v >= t).count();
			
			let timestep_shape = Tensor3Shape { // shape at each timestep
						dim1: n_seqs_included as c_int,
						dim2: self.vec_sz,
						dim3: 1
			};
			//println!("t {} seq_len {}", t, n_seqs_included);
			let tensor_desc = TensorDescriptor::new3(self.data_type, &timestep_shape);
			self.timestep_tensor_desc_vals.push(tensor_desc.val);
			self.timestep_tensor_descs.push(tensor_desc);
		}
	}
	
	// each batch example may have a sequence different than all other batch examples,
	// but it should not exceed max_seq_len
	pub fn update_valid_tpoints(&mut self, seq_len_array: &Vec<c_int>, max_seq_len: c_int) {
		assert!(self.mem.n_elements >= (max_seq_len*self.batch_sz*self.vec_sz) as usize,
				"insufficient gpu memory allocated for sequence {} -> {}, max_seq_len {} batch_sz {} vec_sz {}",
				self.mem.n_elements, max_seq_len*self.batch_sz*self.vec_sz, max_seq_len, self.batch_sz, self.vec_sz);
		
		self.max_seq_len = max_seq_len;
		
		self.desc = RNNDataDescriptor::new(self.data_type, RNN_DATA_LAYOUT, max_seq_len, self.batch_sz, self.vec_sz, seq_len_array);
		self.seq_len_array = seq_len_array.clone();
		
		// for layers which do not deal with time, ravel the batch dimension to include the time dimension too
		self.ravel_time_shape.n = max_seq_len * self.batch_sz;
		self.ravel_time_tensor_desc = TensorDescriptor::new(self.data_type, cudnnTensorFormat_t::CUDNN_TENSOR_NCHW, self.ravel_time_shape);
		
		self.update_timestep_tensor_descs();
	}
	
	pub fn zeros(data_type: cudnnDataType_t, max_seq_len: c_int, batch_sz: c_int,
			vec_sz: c_int, seq_len_array: &Vec<c_int>) -> Self {
		debug_assert!(data_type == cudnnDataType_t::CUDNN_DATA_FLOAT);
		
		let rnn_data = Self::new(data_type, max_seq_len, batch_sz, vec_sz, seq_len_array);
		rnn_data.zero_out();
		rnn_data
	}
	
	pub fn zero_out(&self) {
		self.mem.zero_out();
	}
	
	pub fn one_out(&self) {
		self.mem.one_out();
	}
	
	pub fn ret(&self) -> Vec<f32> {
		self.mem.ret((self.max_seq_len * self.batch_sz * self.vec_sz) as size_t)
	}
	
	pub fn set<T>(&self, src: &Vec<T>) {
		self.mem.set(src);
	}
	
	pub fn n_elements(&self) -> usize {
		self.mem.n_elements
	}
}

impl Filter3 {
	pub fn new(data_type: cudnnDataType_t, sz: size_t) -> Self {
		debug_assert!(TENSOR_FORMAT == cudnnTensorFormat_t::CUDNN_TENSOR_NCHW);
		Self {
			desc: FilterDescriptor::new3(data_type, TENSOR_FORMAT, sz as c_int),
			tensor_desc: TensorDescriptor::new(data_type, TENSOR_FORMAT, TensorShape {n: sz as c_int, c: 1, h: 1, w: 1}),
			mem: gpuMem::new(data_type, sz),
			shape: FilterShape {k: sz as c_int, c: 1, h: 1, w: 1}
		}
	}
	
	pub fn zeros(data_type: cudnnDataType_t, sz: size_t) -> Self {
		debug_assert!(data_type == cudnnDataType_t::CUDNN_DATA_FLOAT);
		
		let filter = Self::new(data_type, sz);
		
		// set values
		let vals: Vec<f32> = vec!{0.; sz};
		filter.mem.set(&vals);
		
		filter
	}
	
	pub fn ones(data_type: cudnnDataType_t, sz: size_t) -> Self {
		debug_assert!(data_type == cudnnDataType_t::CUDNN_DATA_FLOAT);
		
		let filter = Self::new(data_type, sz);
		
		// set values
		let vals: Vec<f32> = vec!{1.; sz};
		filter.mem.set(&vals);
		
		filter
	}
	
	pub fn new_norm(data_type: cudnnDataType_t, sz: size_t,
			norm_scale: f32, rng: &mut XorState) -> Self {
		debug_assert!(data_type == cudnnDataType_t::CUDNN_DATA_FLOAT);
		
		let filter = Self::new(data_type, sz);
		
		// set values
		let vals = rng.gen_norm_vec(sz, norm_scale);
		filter.mem.set(&vals);
		
		filter
	}

	pub fn zero_out(&self) {
		unsafe {cudaMemset(self.mem.val, 0, self.mem.bytes)}.chk_err();
	}
	
	pub fn ret(&self) -> Vec<f32> {
		self.mem.ret(self.shape.n_elements())
	}
}

impl Filter {
	pub fn new(data_type: cudnnDataType_t, shape: FilterShape) -> Self {
		debug_assert!(TENSOR_FORMAT == cudnnTensorFormat_t::CUDNN_TENSOR_NCHW);
		Self {
			desc: FilterDescriptor::new(data_type, TENSOR_FORMAT, shape),
			tensor_desc: TensorDescriptor::new(data_type, TENSOR_FORMAT, TensorShape::from(&shape)),
			mem: gpuMem::new(data_type, shape.n_elements()),
			shape
		}
	}
	
	pub fn zeros(data_type: cudnnDataType_t, shape: FilterShape) -> Self {
		let filter = Self::new(data_type, shape);
		
		// set values
		match data_type {
			cudnnDataType_t::CUDNN_DATA_FLOAT => {
				let vals: Vec<f32> = vec!{0.; shape.n_elements()};
				filter.mem.set(&vals);
			} cudnnDataType_t::CUDNN_DATA_HALF => {
				let vals: Vec<f16> = vec!{0; shape.n_elements()};
				filter.mem.set(&vals);
			} _ => {panic!("unsupported datatype");}
		}
		
		filter
	}
	
	pub fn ones(data_type: cudnnDataType_t, shape: FilterShape) -> Self {
		debug_assert!(data_type == cudnnDataType_t::CUDNN_DATA_FLOAT);
		
		let filter = Self::new(data_type, shape);
		
		// set values
		let vals: Vec<f32> = vec!{1.; shape.n_elements()};
		filter.mem.set(&vals);
		
		filter
	}
	
	// w/ WeightInitialization
	pub fn new_init(data_type: cudnnDataType_t, shape: FilterShape, weight_initialization: WeightInitialization,
			rng: &mut XorState) -> Self {
		match weight_initialization {
			WeightInitialization::NormScale(norm_scale) => {Self::new_norm(data_type, shape, norm_scale, rng)}
			WeightInitialization::XavierUniform(fan_out, fan_in) => {
				let norm_scale = (6./(fan_out + fan_in) as f32).sqrt();
				Self::new_uniform(data_type, shape, norm_scale, rng)
			}
		}
	}
	
	pub fn new_norm(data_type: cudnnDataType_t, shape: FilterShape,
			norm_scale: f32, rng: &mut XorState) -> Self {
		let filter = Self::new(data_type, shape);
		
		// set values
		match data_type {
			cudnnDataType_t::CUDNN_DATA_FLOAT => {
				let vals = rng.gen_norm_vec(shape.n_elements(), norm_scale);
				filter.mem.set(&vals);
			} cudnnDataType_t::CUDNN_DATA_HALF => {
				let vals = rng.gen_norm_vec_f16(shape.n_elements(), norm_scale);
				filter.mem.set(&vals);
			} _ => {panic!("unsupported datatype");}
		}
		
		filter
	}
	
	// generate values in range [-norm_scale, norm_scale]
	pub fn new_uniform(data_type: cudnnDataType_t, shape: FilterShape, norm_scale: f32, rng: &mut XorState) -> Self {
		debug_assert!(data_type == cudnnDataType_t::CUDNN_DATA_FLOAT);
		
		let filter = Filter::new(data_type, shape);
		
		// set values
		let vals = rng.gen_vec(shape.n_elements(), norm_scale);
		filter.mem.set(&vals);
		
		filter
	}
	
	pub fn zero_out(&self) {
		unsafe {cudaMemset(self.mem.val, 0, self.mem.bytes)}.chk_err();
	}
	
	pub fn ret(&self) -> Vec<f32> {
		self.mem.ret(self.shape.n_elements())
	}
}

pub fn prod(dims: &Vec<c_int>) -> c_int {
	let mut p = 1;
	for dim in dims {
		p *= dim;
	}
	p
}

// used w/ einsum
pub struct WorkspaceWShape {
	mem: gpuMem_t,
	shape: Vec<c_int>
}

impl gpuMem {
	pub fn w_shape(&self, shape: Vec<c_int>) -> WorkspaceWShape {
		assert!(self.n_elements >= prod(&shape) as usize);
		
		WorkspaceWShape {
			mem: self.val,
			shape
		}
	}
}

// used w/ einsum
pub trait MemWShape {
	fn mem(&self) -> gpuMem_t;
	fn shape(&self) -> Vec<c_int>;
}

impl MemWShape for &Tensor {
	fn mem(&self) -> gpuMem_t {self.mem.val}
	
	fn shape(&self) -> Vec<c_int> {
		let mut s = Vec::with_capacity(4);
		
		s.push(self.shape.n);
		if self.shape.c == 1 && self.shape.h == 1 && self.shape.w == 1 {return s;}
		s.push(self.shape.c);
		if self.shape.h == 1 && self.shape.w == 1 {return s;}
		s.push(self.shape.h);
		if self.shape.w == 1 {return s;}
		s.push(self.shape.w);
		
		s
	}
}

impl MemWShape for Filter {
	fn mem(&self) -> gpuMem_t {self.mem.val}
	
	fn shape(&self) -> Vec<c_int> {
		let mut s = Vec::with_capacity(4);
		
		s.push(self.shape.k);
		if self.shape.c == 1 && self.shape.h == 1 && self.shape.w == 1 {return s;}
		s.push(self.shape.c);
		if self.shape.h == 1 && self.shape.w == 1 {return s;}
		s.push(self.shape.h);
		if self.shape.w == 1 {return s;}
		s.push(self.shape.w);
		
		s
	}
}

impl MemWShape for WorkspaceWShape {
	fn mem(&self) -> gpuMem_t {self.mem}
	fn shape(&self) -> Vec<c_int> {self.shape.clone()}
}

