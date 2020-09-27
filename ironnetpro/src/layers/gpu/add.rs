use super::*;
use std::ffi::c_void;
use std::cmp::min;
use crate::saving::{KeyPair, find_req_key_parse};

// Forward computes:
// 	y = alpha1*x1 + alpha2*x2
// Where x1 is the previous layer, and the supplied layer2_ind to model.add_add() is the x2 layer

#[allow(non_camel_case_types)]
type DATA_TYPE = f32;

pub struct ReduceData {
	pub desc: ReduceTensorDescriptor,
	pub indices: gpuMem,
	pub workspace: gpuMem
}

pub struct AddInternals {
	params: AddParams,
	op_tensor_desc: OpTensorDescriptor,
	reduce_data: Option<ReduceData>,
	alpha1: Vec<DATA_TYPE>,
	alpha2: Vec<DATA_TYPE>,
}

impl ReduceData {
	pub fn new(handle: &Handle, x_desc: &TensorDescriptor, dy_desc: &TensorDescriptor, data_type: cudnnDataType_t) -> Self {
		let desc = ReduceTensorDescriptor::new(
				cudnnReduceTensorOp_t::CUDNN_REDUCE_TENSOR_ADD,
				data_type, NAN_PROP,
				cudnnReduceTensorIndices_t::CUDNN_REDUCE_TENSOR_NO_INDICES,
				cudnnIndicesType_t::CUDNN_32BIT_INDICES);
		
		let mut workspace_sz = 0;
		let mut indices_sz = 0;
		
		unsafe {cudnnGetReductionWorkspaceSize(handle.cudnn_val, desc.val, x_desc.val, dy_desc.val, &mut workspace_sz)}.chk_err();
		unsafe {cudnnGetReductionIndicesSize(handle.cudnn_val, desc.val, x_desc.val, dy_desc.val, &mut indices_sz)}.chk_err();
		
		let workspace = gpuMem::new(cudnnDataType_t::CUDNN_DATA_INT8, workspace_sz);
		let indices = gpuMem::new(cudnnDataType_t::CUDNN_DATA_INT8, indices_sz);
		// ^ fails w/ CUDNN_STATUS_INVALID_VALUE when workspace_sz & indices_sz are allocated, indicating insufficient workspace sizes
		
		Self {desc, indices, workspace}
	}
}

impl Run for AddInternals {
	fn forward(&self, layer: &Layer, model: &Model) {
		#[cfg(feature="profile")]
		let _g = Guard::new("add fwd");

		debug_assert!(layer.x_layers.len() == 2);
		
		let x1 = model.layers[layer.x_layers[0]].y.tensor(); // output of input layer is the input for this layer
		let x2 = model.layers[layer.x_layers[1]].y.tensor();
		let y = layer.y.tensor();
		
		/*println!("alpha1 {} alpha2 {} x1 {} x2 {} layer inds {}, {} layer nms: {}, {}",
				self.alpha1[0], self.alpha2[0],
				x1.shape.to_string(), x2.shape.to_string(),
				layer.x_layers[0], layer.x_layers[1],
				model.layers[layer.x_layers[0]].nm, model.layers[layer.x_layers[1]].nm);*/
		
		unsafe {cudnnOpTensor(model.handle.cudnn_val,
				self.op_tensor_desc.val,
				
				self.alpha1.as_ptr() as *const c_void,
				x1.desc.val, x1.mem.val,
				
				self.alpha2.as_ptr() as *const c_void,
				x2.desc.val, x2.mem.val,
				
				model.zero(layer.data_type),
				y.desc.val, y.mem.val)}.chk_err();
	}
	
	fn backward(&self, layer: &Layer, model: &Model) {
		#[cfg(feature="profile")]
		let _g = Guard::new("add bwd");

		let dx1 = model.layers[layer.x_layers[0]].dy.tensor();
		let dx2 = model.layers[layer.x_layers[1]].dy.tensor();
		let dy = layer.dy.tensor();
		
		// input 1
		// dx1 += dy*alpha1
		
		unsafe {cudnnAddTensor(model.handle.cudnn_val,
				self.alpha1.as_ptr() as *const c_void,
				dy.desc.val, dy.mem.val,
				
				model.one(layer.data_type),
				dx1.desc.val, dx1.mem.val)}.chk_err();
		
		//println!("add backward x1; dy shape {} dx1 shape {}, alpha1 {} dy {} dx1 {} {}",
		//		dy.shape.to_string(), dx1.shape.to_string(), self.alpha1[0],
		//		dy.ret()[0], dx1.ret()[0], layer.x_layers[0]);
		
		if let Some(reduce_data) = &self.reduce_data {
				//println!("add backward dy {} dx2 {} {}", dy.shape.to_string(), dx2.shape.to_string(),
				//		reduce_data.workspace.bytes);
				
				unsafe {cudnnReduceTensor(model.handle.cudnn_val,
					reduce_data.desc.val,
					reduce_data.indices.val, reduce_data.indices.bytes,
					reduce_data.workspace.val, reduce_data.workspace.bytes,
					
					self.alpha2.as_ptr() as *const c_void,
					dy.desc.val, dy.mem.val,
					
					model.one(layer.data_type),
					dx2.desc.val, dx2.mem.val)}.chk_err();
		}else{
				unsafe {cudnnAddTensor(model.handle.cudnn_val,
					self.alpha2.as_ptr() as *const c_void,
					dy.desc.val, dy.mem.val,
					
					model.one(layer.data_type),
					dx2.desc.val, dx2.mem.val)}.chk_err();
		}
	}
	
	fn sv_arch(&self, txt: &mut String) {
		txt.push_str(&format!("\talpha1: {}\n", self.params.alpha1));
		txt.push_str(&format!("\talpha2: {}\n", self.params.alpha2));
		txt.push_str(&format!("\tdata_type: {}\n", self.params.data_type));
	}
	
	fn workspace_sz(&self) -> usize {
		if let Some(reduce_data) = &self.reduce_data {
			reduce_data.indices.bytes + 
			reduce_data.workspace.bytes
		}else {0}
	}
}

#[derive(Copy, Clone)]
pub struct AddParams {
	pub alpha1: DATA_TYPE,
	pub alpha2: DATA_TYPE,
	pub data_type: cudnnDataType_t
}

impl Model {
	pub fn add_add(&mut self, layer2_ind: usize, params: AddParams) {
		let data_type = params.data_type;
		let alpha1 = params.alpha1;
		let alpha2 = params.alpha2;
		debug_assert!(self.layers.len() > 0);
		debug_assert!(params.data_type == cudnnDataType_t::CUDNN_DATA_FLOAT);
		
		let layer_prev_ind = self.layers.len() - 1;
		debug_assert!(layer2_ind != layer_prev_ind);
		
		let x1 = self.layers[layer_prev_ind].y.tensor(); // (input to this layer is the output of the previous layer)
		let dx1 = self.layers[layer_prev_ind].dy.tensor();
		
		let x2 = self.layers[layer2_ind].y.tensor();
		let dx2 = self.layers[layer2_ind].dy.tensor();
		
		debug_assert!(x1.mem.dataType == x2.mem.dataType && x1.mem.dataType == data_type);
		debug_assert!(data_type.bytes() == size_of::<DATA_TYPE>(), "alpha, beta for add op should be changed or generalized");
		
		let op_tensor_desc = OpTensorDescriptor::new(cudnnOpTensorOp_t::CUDNN_OP_TENSOR_ADD, data_type, NAN_PROP);
		
		let y_shape = TensorShape::broadcast(x1.shape, x2.shape);
		let y = Tensor::new(data_type, y_shape);
		let dy = Tensor::new(data_type, y_shape);
		
		// cudnnOpTensor expects the second tensor to be the smaller, if broadcasting
		// is to be performed (ie. x1: (3,4,5,6), x2: (3,1,1,6) is valid but
		// reversing the input parameters (x1 as x2 and x2 as x1) results in an error)
		let layer = if x1.shape.n > x2.shape.n || x1.shape.c > x2.shape.c ||
		   		   x1.shape.h > x2.shape.h || x1.shape.w > x2.shape.w {
			debug_assert!(x1.shape.n >= x2.shape.n && x1.shape.c >= x2.shape.c &&
					  x1.shape.h >= x2.shape.h && x1.shape.w >= x2.shape.w);
			
			// if we are broadcasting, we have to sum dy into dx2
			let reduce_data = if x1.shape != x2.shape {
				Some(ReduceData::new(&self.handle, &dx2.desc, &dy.desc, data_type))
			}else {None};
			
			Layer::new_w_dy(
				vec![layer_prev_ind, layer2_ind],
				InternalTypes::Add(AddInternals {
						params,
						op_tensor_desc,
						reduce_data,
						alpha1: vec![alpha1],
						alpha2: vec![alpha2],
				}),
				y, dy,
				String::from("add"),
				data_type
			)
		}else{
			debug_assert!(x2.shape.n >= x1.shape.n && x2.shape.c >= x1.shape.c &&
					  x2.shape.h >= x1.shape.h && x2.shape.w >= x1.shape.w);
			
			// if we are broadcasting, we have to sum dy into dx1
			let reduce_data = if x1.shape != x2.shape {
				Some(ReduceData::new(&self.handle, &dy.desc, &dx1.desc, data_type))
			}else {None};
			 
			Layer::new_w_dy(
				vec![layer2_ind, layer_prev_ind],
				InternalTypes::Add(AddInternals {
						params,
						op_tensor_desc,
						reduce_data,
						alpha1: vec![alpha2],
						alpha2: vec![alpha1],
				}),
				y, dy,
				String::from("add"),
				data_type
			)
		}; 
		
		self.layers.push(layer);
	}
	
	pub fn load_add(&mut self, x_layers: &Vec<usize>, layer_keys: &Vec<KeyPair>) {
		let layer2_ind = min(x_layers[0], x_layers[1]);
		self.add_add(layer2_ind, AddParams {
			alpha1: find_req_key_parse("alpha1", layer_keys),
			alpha2: find_req_key_parse("alpha2", layer_keys),
			data_type: find_req_key_parse("data_type", layer_keys)
		});
	}
}

