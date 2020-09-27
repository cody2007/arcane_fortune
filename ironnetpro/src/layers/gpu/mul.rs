use super::*;
use crate::saving::{KeyPair, find_req_key_parse};
use std::cmp::min;
use crate::layers::MulParams;

pub struct ReduceVars {
	pub reduction_workspaces: ReductionWorkspaces,
	pub dx2_tmp: Tensor
}

pub struct MulInternals {
	params: MulParams,
	op_tensor_desc: OpTensorDescriptor, // for the multiplication
	reduce_vars: Option<ReduceVars> // if a reduction needs to be computed for the gradient computation
}

impl Run for MulInternals {
	fn forward(&self, layer: &Layer, model: &Model) {
		#[cfg(feature="profile")]
		let _g = Guard::new("mul fwd");
		
		let x1 = &model.layers[layer.x_layers[0]].y; // output of input layer is the input for this layer
		let x2 = &model.layers[layer.x_layers[1]].y;
		let y = &layer.y;
		
		//println!("fwd x1 {} x2 {} y {}", x1.ravel_time_shape().to_string(), 
		//		x2.ravel_time_shape().to_string(),
		//		y.ravel_time_shape().to_string());
		
		unsafe {cudnnOpTensor(model.handle.cudnn_val,
				self.op_tensor_desc.val,
				
				model.one(layer.data_type),
				x1.ravel_time_tensor_desc(), x1.mem(),
				
				model.one(layer.data_type),
				x2.ravel_time_tensor_desc(), x2.mem(),
				
				model.zero(layer.data_type),
				y.ravel_time_tensor_desc(), y.mem())}.chk_err();
	}
	
	fn backward(&self, layer: &Layer, model: &Model) {
		#[cfg(feature="profile")]
		let _g = Guard::new("mul bwd");

		let layer1 = &model.layers[layer.x_layers[0]];
		let layer2 = &model.layers[layer.x_layers[1]];
		
		let x1 = &layer1.y;
		let x2 = &layer2.y;
		
		let dx1 = &layer1.dy;
		let dx2 = &layer2.dy;
		
		let dy = &layer.dy;
		
		// product rule: f(x) = x1*x2 ;   df/dx = x1*x2' + x1'*x2
		
		// input 1
		// dx1 += dy*x2
		// dx1 = (dy * x2) + dx1
		
		/*println!("mul bwd dy {} x2 {} x1 {} dx2 {} dx1 {}", dy.ravel_time_shape().to_string(),
				x2.ravel_time_shape().to_string(),
				x1.ravel_time_shape().to_string(),
				dx2.ravel_time_shape().to_string(),
				dx1.ravel_time_shape().to_string());*/
		
		unsafe {cudnnOpTensor(model.handle.cudnn_val,
				self.op_tensor_desc.val,
				
				model.one(layer.data_type),
				dy.ravel_time_tensor_desc(), dy.mem(),
				
				model.one(layer.data_type),
				x2.ravel_time_tensor_desc(), x2.mem(),
				
				model.one(layer.data_type),
				dx1.ravel_time_tensor_desc(), dx1.mem())}.chk_err();
		
		// need to sum reduce dy*x1 to get to dx2's shape
		if let Some(reduce_vars) = &self.reduce_vars { 
			// dx2_tmp = dy*x1
			unsafe {cudnnOpTensor(model.handle.cudnn_val,
				self.op_tensor_desc.val,
				
				model.one(layer.data_type),
				dy.ravel_time_tensor_desc(), dy.mem(),
				
				model.one(layer.data_type),
				x1.ravel_time_tensor_desc(), x1.mem(),
				
				model.zero(layer.data_type),
				reduce_vars.dx2_tmp.desc.val, reduce_vars.dx2_tmp.mem.val)}.chk_err();
			
			// dx2 += dx2_tmp
			unsafe {cudnnReduceTensor(model.handle.cudnn_val,
				reduce_vars.reduction_workspaces.desc.val,
				reduce_vars.reduction_workspaces.indices.val, reduce_vars.reduction_workspaces.indices.bytes,
				reduce_vars.reduction_workspaces.workspace.val, reduce_vars.reduction_workspaces.workspace.bytes,
				
				model.one(layer.data_type),
				reduce_vars.dx2_tmp.desc.val, reduce_vars.dx2_tmp.mem.val,
				
				model.one(layer.data_type),
				dx2.ravel_time_tensor_desc(), dx2.mem())}.chk_err();
		}else{
			unsafe {cudnnOpTensor(model.handle.cudnn_val,
				self.op_tensor_desc.val,
				
				model.one(layer.data_type),
				dy.ravel_time_tensor_desc(), dy.mem(),
				
				model.one(layer.data_type),
				x1.ravel_time_tensor_desc(), x1.mem(),
				
				model.one(layer.data_type),
				dx2.ravel_time_tensor_desc(), dx2.mem())}.chk_err();
		}
	}
	
	fn sv_arch(&self, txt: &mut String) {
		txt.push_str(&format!("\tdata_type: {}\n", self.params.data_type));
	}
	
	fn workspace_sz(&self) -> usize {
		if let Some(reduce_vars) = &self.reduce_vars {
			reduce_vars.reduction_workspaces.indices.bytes +
			reduce_vars.reduction_workspaces.workspace.bytes + 
			reduce_vars.dx2_tmp.mem.bytes
		}else {0}
	}
}

impl Model {
	// multiplies the most recently added layer with layer2_ind
	pub fn add_mul(&mut self, layer2_ind: usize, params: MulParams) {
		let data_type = params.data_type;
		debug_assert!(self.layers.len() > 0);
		
		let layer_prev_ind = self.layers.len() - 1;
		
		debug_assert!(layer_prev_ind != layer2_ind);
		let x1 = &self.layers[layer_prev_ind].y; // (input to this layer is the output of the previous layer)
		let x2 = &self.layers[layer2_ind].y;
		
		let x1_shape = x1.ravel_time_shape();
		let x2_shape = x2.ravel_time_shape();
		
		debug_assert!(x1.data_type() == x2.data_type() && x1.data_type() == data_type);
		
		let op_tensor_desc = OpTensorDescriptor::new(cudnnOpTensorOp_t::CUDNN_OP_TENSOR_MUL, cudnnDataType_t::CUDNN_DATA_FLOAT,
				NAN_PROP); //data_type, NAN_PROP);
		
		let y_shape = TensorShape::broadcast(x1_shape, x2_shape);
		let y = Tensor::new(data_type, y_shape);
		
		// cudnnOpTensor expects the second tensor to be the smaller, if broadcasting
		// is to be performed (ie. x1: (3,4,5,6), x2: (3,1,1,6) is valid but
		// reversing the input parameters (x1 as x2 and x2 as x1) results in an error)
		let layer = if x1_shape.n > x2_shape.n || x1_shape.c > x2_shape.c ||
				   x1_shape.h > x2_shape.h || x1_shape.w > x2_shape.w {
			debug_assert!(x1_shape.n >= x2_shape.n && x1_shape.c >= x2_shape.c &&
					  x1_shape.h >= x2_shape.h && x1_shape.w >= x2_shape.w);
			Layer::new(
				vec![layer_prev_ind, layer2_ind],
				InternalTypes::Mul(MulInternals {params, op_tensor_desc, reduce_vars: None}),
				y,
				String::from("mul"),
				data_type
			)
		}else{
			debug_assert!(x2_shape.n >= x1_shape.n && x2_shape.c >= x1_shape.c &&
					  x2_shape.h >= x1_shape.h && x2_shape.w >= x1_shape.w);
			//println!("mul ordering {} {}", layer2_ind, layer_prev_ind);
			
			let reduce_vars = if x1_shape != x2_shape {
				Some(ReduceVars {
					reduction_workspaces: ReductionWorkspaces::new(self, &x2.tensor().desc, &x1.tensor().desc),
					dx2_tmp: Tensor::new(data_type, x2_shape)
				})
			}else {None};
			
			Layer::new(
				vec![layer2_ind, layer_prev_ind],
				InternalTypes::Mul(MulInternals {params, op_tensor_desc, reduce_vars}),
				y,
				String::from("mul"),
				data_type
			)
		};
		
		self.layers.push(layer);
	}
	
	pub fn load_mul(&mut self, x_layers: &Vec<usize>, layer_keys: &Vec<KeyPair>) {
		let layer2_ind = min(x_layers[0], x_layers[1]);
		// ^ layer2_ind: the earlier layer, the function assumes the later layer is the layer that was added right before this one
		self.add_mul(layer2_ind, MulParams {
				data_type: find_req_key_parse("data_type", layer_keys)
		});
	}
}

