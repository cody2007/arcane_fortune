#![allow(non_snake_case)]
use super::*;
use crate::saving::{KeyPair, find_req_key_parse, find_key_vec};
use crate::layers::ElementwiseAffineParams;
use crate::layers::gpu::ReduceVars;

pub struct ElementwiseAffineInternals {
	params: ElementwiseAffineParams,
	reduce_vars: ReduceVars,
	
	mul_op_tensor_desc: OpTensorDescriptor,
	add_op_tensor_desc: OpTensorDescriptor,
	
	pub gain: Filter,
	pub dgain: Filter,
	
	pub bias: Filter,
	pub dbias: Filter
}

impl Run for ElementwiseAffineInternals {
	fn forward(&self, layer: &Layer, model: &Model) {
		#[cfg(feature="profile")]
		let _g = Guard::new("elementwise affine fwd");
		
		debug_assert!(layer.x_layers.len() == 1);
		let x = &model.layers[layer.x_layers[0]].y; // output of input layer is the input for this layer
		let y = &layer.y;
		
		let y_tmp = model.shared_workspace.as_ref().unwrap();
		
		// y_tmp = gain*x
		unsafe {cudnnOpTensor(model.handle.cudnn_val,
				self.mul_op_tensor_desc.val,
				
				model.one(layer.data_type),
				x.ravel_time_tensor_desc(), x.mem(),
				
				model.one(layer.data_type),
				self.gain.tensor_desc.val, self.gain.mem.val,
				
				model.zero(layer.data_type),
				y.ravel_time_tensor_desc(), y_tmp.val)}.chk_err();
		
		// y = y_tmp + bias
		unsafe {cudnnOpTensor(model.handle.cudnn_val,
				self.add_op_tensor_desc.val,
				
				model.one(layer.data_type),
				y.ravel_time_tensor_desc(), y_tmp.val,
				
				model.one(layer.data_type),
				self.bias.tensor_desc.val, self.bias.mem.val,
				
				model.zero(layer.data_type),
				y.ravel_time_tensor_desc(), y.mem())}.chk_err();
	}
	
	fn backward(&self, layer: &Layer, model: &Model) {
		#[cfg(feature="profile")]
		let _g = Guard::new("elementwise affine bwd");
		
		debug_assert!(layer.x_layers.len() == 1);
		let x = &model.layers[layer.x_layers[0]].y; // output of input layer is the input for this layer
		let dx = &model.layers[layer.x_layers[0]].dy;
		let dy = &layer.dy;
		
		// dbias += dy (need to sum reduce dy to get to dbias' shape)
		unsafe {cudnnReduceTensor(model.handle.cudnn_val,
				self.reduce_vars.reduction_workspaces.desc.val,
				self.reduce_vars.reduction_workspaces.indices.val,
				self.reduce_vars.reduction_workspaces.indices.bytes,
				self.reduce_vars.reduction_workspaces.workspace.val,
				self.reduce_vars.reduction_workspaces.workspace.bytes,
				
				model.one(layer.data_type),
				dy.ravel_time_tensor_desc(), dy.mem(),
				
				model.one(layer.data_type),
				self.dbias.tensor_desc.val, self.dbias.mem.val)}.chk_err();
		
		// dgain += dy*x (first multiply dy*dx, then sum reduce to get to dgain's shape)
		{
			unsafe {cudnnOpTensor(model.handle.cudnn_val,
				self.mul_op_tensor_desc.val,
				
				model.one(layer.data_type),
				dy.ravel_time_tensor_desc(), dy.mem(),
				
				model.one(layer.data_type),
				x.ravel_time_tensor_desc(), x.mem(),
				
				model.zero(layer.data_type),
				self.reduce_vars.dx2_tmp.desc.val, self.reduce_vars.dx2_tmp.mem.val)}.chk_err();
			
			unsafe {cudnnReduceTensor(model.handle.cudnn_val,
				self.reduce_vars.reduction_workspaces.desc.val,
				self.reduce_vars.reduction_workspaces.indices.val,
				self.reduce_vars.reduction_workspaces.indices.bytes,
				self.reduce_vars.reduction_workspaces.workspace.val,
				self.reduce_vars.reduction_workspaces.workspace.bytes,
				
				model.one(layer.data_type),
				self.reduce_vars.dx2_tmp.desc.val, self.reduce_vars.dx2_tmp.mem.val,
				
				model.one(layer.data_type),
				self.dgain.tensor_desc.val, self.dgain.mem.val)}.chk_err();
		}
		
		// dx += dy*gain
		unsafe {cudnnOpTensor(model.handle.cudnn_val,
				self.mul_op_tensor_desc.val,
				
				model.one(layer.data_type),
				dy.ravel_time_tensor_desc(), dy.mem(),
				
				model.one(layer.data_type),
				self.gain.tensor_desc.val, self.gain.mem.val,
				
				model.one(layer.data_type),
				dx.ravel_time_tensor_desc(), dx.mem())}.chk_err();
	}
	
	fn zero_out_internal_gradients(&self) {
		self.dgain.zero_out();
		self.dbias.zero_out();
	}
	
	fn gradients(&self) -> Vec<Weights> {
		vec![Weights {
			w_desc: self.gain.tensor_desc.val, 
			w_mem: self.gain.mem.val,
			dw_desc: self.dgain.tensor_desc.val,
			dw_mem: self.dgain.mem.val,
			len: self.gain.mem.n_elements,
			data_type: self.gain.mem.dataType
		    },
		    Weights {
		      w_desc: self.bias.tensor_desc.val, 
			w_mem: self.bias.mem.val,
			dw_desc: self.dbias.tensor_desc.val,
			dw_mem: self.dbias.mem.val,
			len: self.bias.mem.n_elements,
			data_type: self.bias.mem.dataType
		    }]
	}
	
	fn sv_arch(&self, txt: &mut String) {
		txt.push_str(&format!("\tdata_type: {}\n", self.params.data_type));
		txt.push_str("\tdims: ");
		for (dim_id, dim) in self.params.dims.iter().enumerate() {
			txt.push_str(&format!("{}", dim));
			if dim_id != (self.params.dims.len() - 1) {
				txt.push_str(", ");
			}
		}
		txt.push_str("\n");
	}
	
	fn sv_weights(&self, _: &Layer, _: &Model, save_dir: &str, file_nm: &str) {
		self.gain.sv(save_dir, &format!("{}_gain", file_nm));
		self.bias.sv(save_dir, &format!("{}_bias", file_nm));
	}
	
	fn sv_gradients(&self, save_dir: &str, file_nm: &str) {
		self.dgain.sv(save_dir, &format!("{}_gain", file_nm));
		self.dbias.sv(save_dir, &format!("{}_bias", file_nm));
	}
	
	fn ld_weights(&self, _: &Layer, _: &Model, save_dir: &str, file_nm: &str) {
		self.gain.ld(save_dir, &format!("{}_gain", file_nm));
		self.bias.ld(save_dir, &format!("{}_bias", file_nm));
	}
	
	fn workspace_sz(&self) -> usize {
		self.reduce_vars.reduction_workspaces.indices.bytes +
		self.reduce_vars.reduction_workspaces.workspace.bytes + 
		self.reduce_vars.dx2_tmp.mem.bytes
	}
}

impl Model {
	// y = gain*x + bias, where everything has the same shape
	pub fn add_elementwise_affine(&mut self, params: ElementwiseAffineParams) {
		let data_type = params.data_type;
		println!("dims {:?}", params.dims);
		debug_assert!(self.layers.len() > 0);
		
		let layer_prev_ind = self.layers.len() - 1;
		
		let x = &self.layers[layer_prev_ind].y; // (input to this layer is the output of the previous layer)
		
		let x_shape = x.ravel_time_shape();
		
		debug_assert!(x.data_type() == data_type);
		debug_assert!(data_type == cudnnDataType_t::CUDNN_DATA_FLOAT);
		
		let mul_op_tensor_desc = OpTensorDescriptor::new(cudnnOpTensorOp_t::CUDNN_OP_TENSOR_MUL, cudnnDataType_t::CUDNN_DATA_FLOAT, NAN_PROP);
		let add_op_tensor_desc = OpTensorDescriptor::new(cudnnOpTensorOp_t::CUDNN_OP_TENSOR_ADD, cudnnDataType_t::CUDNN_DATA_FLOAT, NAN_PROP);
		
		// used w/ forward because it cannot work in-place w/ same input & output buffer
		self.allocate_shared_workspace(params.data_type, x_shape.n_elements());
		let x = &self.layers[layer_prev_ind].y;
		
		let y = Tensor::new(data_type, x_shape);
		
		// chk params.dims
		{
			for dim in params.dims.iter() {
				assert!(*dim < 4, "dims out of range: {:?}", params.dims);
			}
			assert!(params.dims.len() <= 4, "too many dims: {:?}", params.dims);
		}
		
		let F_shape = FilterShape {
			k: if params.dims.contains(&0) {x_shape.n} else {1},
			c: if params.dims.contains(&1) {x_shape.c} else {1},
			h: if params.dims.contains(&2) {x_shape.h} else {1},
			w: if params.dims.contains(&3) {x_shape.w} else {1}
		};
		
		let gain = Filter::ones(data_type, F_shape);
		let dgain = Filter::zeros(data_type, F_shape);
		
		let bias = Filter::zeros(data_type, F_shape);
		let dbias = Filter::zeros(data_type, F_shape);
		
		let reduce_vars = {
			let gain_t = Tensor::new(data_type, 
					TensorShape {
						n: F_shape.k,
						c: F_shape.c,
						h: F_shape.h,
						w: F_shape.w
					}
			);
			
			ReduceVars {
				reduction_workspaces: ReductionWorkspaces::new(self, &x.tensor().desc, &gain_t.desc),
				dx2_tmp: Tensor::new(data_type, x_shape)
			}
		};

		self.layers.push(Layer::new(
			vec![layer_prev_ind],
			InternalTypes::ElementwiseAffine(ElementwiseAffineInternals {
					params, reduce_vars,
					mul_op_tensor_desc,
					add_op_tensor_desc,
					gain,
					dgain,
					bias,
					dbias
				}),
			y,
			String::from("elementwise_affine"),
			data_type
		));
	}
	
	pub fn load_elementwise_affine(&mut self, layer_keys: &Vec<KeyPair>) {
		self.add_elementwise_affine(ElementwiseAffineParams {
				data_type: find_req_key_parse("data_type", layer_keys),
				dims: find_key_vec("dims", layer_keys)
		});
	}
}

