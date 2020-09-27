use super::*;
use crate::layers::TransposeReshapeParams;
use std::ffi::c_void;
use crate::saving::{KeyPair, find_req_key_parse};

pub struct TransposeReshapeInternals {
	params: TransposeReshapeParams,
	bwd_dims: Vec<usize>, // for backprop
	y_szs: Vec<size_t>, // number of elements for each dimension of y (before reshaping)
}

fn is_identity(dims: &Vec<usize>) -> bool {
	// [0,1,2,3] -> [0,1,2,3]
	for (pos, dim) in dims.iter().enumerate() {
		if pos != *dim {return false;}
	}
	true
}

impl Run for TransposeReshapeInternals {
	fn forward(&self, layer: &Layer, model: &Model) {
		#[cfg(feature="profile")]
		let _g = Guard::new("transpose reshape fwd");

		debug_assert!(layer.x_layers.len() == 1);
		
		let x = &model.layers[layer.x_layers[0]].y.tensor();
		let y = &layer.y.tensor();
		
		debug_assert!(self.params.data_type == cudnnDataType_t::CUDNN_DATA_FLOAT &&
				  x.mem.dataType == cudnnDataType_t::CUDNN_DATA_FLOAT &&
				  y.mem.dataType == cudnnDataType_t::CUDNN_DATA_FLOAT);
		
		debug_assert!(x.mem.bytes == y.mem.bytes);
		
		// identity, only a reshaping has occured, just copy data
		if is_identity(&self.bwd_dims) {
			unsafe {cudaMemcpy(y.mem.val as *mut c_void,
						 x.mem.val as *const c_void,
						 y.mem.bytes,
						 cudaMemcpyKind::cudaMemcpyDeviceToDevice)}.chk_err();
		// re-arrange data
		}else{
			unsafe {transpose(self.params.fwd_dims[0], self.params.fwd_dims[1],
						  self.params.fwd_dims[2], self.params.fwd_dims[3],
						  x.shape.n as size_t, x.shape.c as size_t,
						  x.shape.h as size_t, x.shape.w as size_t,
						  x.mem.val, y.mem.val, 0. as f32)}; // x -> y
		}
	}
	
	fn backward(&self, layer: &Layer, model: &Model) {
		#[cfg(feature="profile")]
		let _g = Guard::new("transpose reshape bwd");

		debug_assert!(layer.x_layers.len() == 1);
		
		let dx = &model.layers[layer.x_layers[0]].dy.tensor();
		let dy = &layer.dy.tensor();
		
		debug_assert!(self.params.data_type == cudnnDataType_t::CUDNN_DATA_FLOAT &&
				  dx.mem.dataType == cudnnDataType_t::CUDNN_DATA_FLOAT &&
				  dy.mem.dataType == cudnnDataType_t::CUDNN_DATA_FLOAT);
		
		debug_assert!(dx.mem.bytes == dy.mem.bytes);

		// identity, only a reshaping has occured, just copy data
		if is_identity(&self.bwd_dims) {
			debug_assert!(dy.shape.n_elements() == dx.shape.n_elements());
			// ^ can have some dims raveled together but should have same number of elements
			
			unsafe {cudnnAddTensor(model.handle.cudnn_val,
					model.one(layer.data_type),
					dy.desc.val, dy.mem.val,
					
					model.one(layer.data_type),
					dy.desc.val, dx.mem.val)}.chk_err(); // see note above about using dy.desc 2x
			
		// re-arrange data
		}else{
			unsafe {transpose(self.bwd_dims[0], self.bwd_dims[1],
						  self.bwd_dims[2], self.bwd_dims[3],
						  self.y_szs[0], self.y_szs[1],
						  self.y_szs[2], self.y_szs[3],
						  dy.mem.val, dx.mem.val, 1. as f32)}; // dy -> dx
		}
	}
	
	fn sv_arch(&self, txt: &mut String) {
		for (dim_ind, dim) in self.params.fwd_dims.iter().enumerate() {
			txt.push_str(&format!("\tfwd_dim{}: {}\n", dim_ind, dim));
		}
		txt.push_str(&format!("\tnew_shape: {}\n", self.params.new_shape.to_string()));
		txt.push_str(&format!("\tdata_type: {}\n", self.params.data_type));
	}
}

impl Model {
	// reorders to params.dims, ex [1,2,0,3], which is similar to running np.transpose()
	pub fn add_transpose_reshape(&mut self, params: TransposeReshapeParams) {
		let data_type = params.data_type;
		debug_assert!(self.layers.len() > 0);
		
		let layer_prev = self.layers.len() - 1;
		
		let x_shape = self.layers[layer_prev].y.tensor().shape; // (input to this layer is the output of the previous layer)
				
		// each dim should only appear once and should cover the full range of 0..4
		assert!(params.fwd_dims.len() == 4);
		assert!(params.fwd_dims.iter().sum::<usize>() == (0 + 1 + 2 + 3), "fwd_dims {:?}", params.fwd_dims);
		assert!(params.fwd_dims.iter().min() == Some(&0) && params.fwd_dims.iter().max() == Some(&3));
		
		let y_szs = {
			let x_szs = vec![x_shape.n, x_shape.c, x_shape.h, x_shape.w];
			let mut y_szs = Vec::with_capacity(4);
			for x_dim in params.fwd_dims.iter() {
				y_szs.push(x_szs[*x_dim] as size_t);
			}
			debug_assert!((y_szs[0]*y_szs[1]*y_szs[2]*y_szs[3]) == (x_shape.n_elements() as size_t));
			y_szs
		};
		
		let mut bwd_dims = Vec::with_capacity(4);
		for pos in 0..4 {
			bwd_dims.push(params.fwd_dims.iter().position(|&dim| dim == pos).unwrap());
		}
		
		// can be used to ravel dims, ex: [100, 200, 300*500, 1]  instead of [100, 200, 300, 500]
		let new_shape = params.new_shape;
		assert!(new_shape.n_elements() == x_shape.n_elements());
		//println!("output {}", new_shape.to_string());
		
		self.layers.push( Layer::new(
			vec![layer_prev],
			InternalTypes::TransposeReshape(TransposeReshapeInternals {
				params,
				bwd_dims,
				y_szs
			}),
			Tensor::new(data_type, new_shape),
			String::from("TransposeReshape"),
			data_type
		));
	}
	
	pub fn load_transpose_reshape(&mut self, layer_keys: &Vec<KeyPair>) {
		let mut fwd_dims = Vec::with_capacity(4);
		for dim in 0..4 {
			fwd_dims.push(find_req_key_parse(&format!("fwd_dim{}", dim), layer_keys));
		}
		
		self.add_transpose_reshape(TransposeReshapeParams {
			fwd_dims,
			new_shape: find_req_key_parse("new_shape", layer_keys),
			data_type: find_req_key_parse("data_type", layer_keys)
		});
	}
}
