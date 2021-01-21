use super::*;
use crate::saving::{KeyPair, find_req_key_parse};
use crate::layers::{SumType, SumReduceParams};

pub struct SumReduceInternals {
	params: SumReduceParams,
	reduction_workspaces: ReductionWorkspaces,
}

impl Run for SumReduceInternals {
	fn forward(&self, layer: &Layer, model: &Model) {
		#[cfg(feature="profile")]
		let _g = Guard::new("sum reduce fwd");

		let x = model.layers[layer.x_layers[0]].y.tensor(); // output of input layer is the input for this layer
		let y = layer.y.tensor();
		
		unsafe {cudnnReduceTensor(model.handle.cudnn_val,
				self.reduction_workspaces.desc.val,
				self.reduction_workspaces.indices.val, self.reduction_workspaces.indices.bytes,
				self.reduction_workspaces.workspace.val, self.reduction_workspaces.workspace.bytes,
				
				model.one(layer.data_type),
				x.desc.val, x.mem.val,
				
				model.zero(layer.data_type),
				y.desc.val, y.mem.val)}.chk_err();
	}
	
	fn backward(&self, layer: &Layer, model: &Model) {
		#[cfg(feature="profile")]
		let _g = Guard::new("sum reduce bwd");

		let dx = model.layers[layer.x_layers[0]].dy.tensor();
		let dy = layer.dy.tensor();
		
		// dx += dy
		
		// case of [n,1,1,1] being broadcast added to [n,c,h,w], not supported by cudnnAddTensor
		if dy.shape.n != 1 && dy.shape.c == 1 &&
			dy.shape.h == 1 && dy.shape.w == 1 {
				//println!("sum reduce case 1");
				unsafe {broadcast_across_img_vals(
						dy.mem.val,
						dx.mem.val,
						dx.shape.n_elements() / dx.shape.n as usize, // n_vals per img
						dx.shape.n_elements()
				)};
		
		// case of [n,c,1,1] being broadcast added to [n,c,h,1] seemingly not supported by cudnnAddTensor
		}else if dy.shape.n != 1 && dy.shape.c != 1 &&
			dy.shape.h == 1 && dy.shape.w == 1 {
				//println!("sum reduce case 2 dy {} dx {}", dy.ret()[0], dx.ret()[0]);
				unsafe {broadcast_across_img_vals(
						dy.mem.val,
						dx.mem.val,
						dx.shape.n_elements() / (dx.shape.n * dx.shape.c) as usize, // n_vals per img
						dx.shape.n_elements()
				)};
		// case of [n,1,h,w] (dy) being broadcast added to [n,c,h,w] (dx) seemingly not supported by cudnnAddTensor
		}else if dy.shape.n != 1 && dy.shape.c == 1 &&
			dy.shape.h != 1 && dy.shape.w != 1 {
				
				debug_assert!(dy.shape.n == dx.shape.n);
				debug_assert!(dy.shape.h == dx.shape.h);
				debug_assert!(dy.shape.w == dx.shape.w);
				
				unsafe {broadcast_across_channel_vals(
						dy.mem.val, // input
						dx.mem.val, // output
						dy.shape.n as usize, // imgs
						dx.shape.c as usize, // channels
						(dy.shape.h * dy.shape.w) as usize
				)};
		
		// case of [1,1,1,1] being broadcast added to [n,c,h,w], not supported by cudnnAddTensor
		}else if dy.shape.n == 1 && dy.shape.c == 1 &&
			dy.shape.h == 1 && dy.shape.w == 1 {
				//println!("sum reduce case 3");
				unsafe {broadcast_across_all_vals(
						dy.mem.val,
						dx.mem.val,
						dx.shape.n_elements()
				)};
		}else{
			//println!("sum reduce case 4");
			//println!("dy {}, dx {}", dy.shape.to_string(), dx.shape.to_string());
			unsafe {cudnnAddTensor(model.handle.cudnn_val,
				model.one(layer.data_type),
				dy.desc.val, dy.mem.val,
				
				model.one(layer.data_type),
				dx.desc.val, dx.mem.val)}.chk_err();
			//println!("fin");
		}
	}
	
	fn sv_arch(&self, txt: &mut String) {
		txt.push_str(&format!("\tsum_type: {}\n", self.params.sum_type));
		txt.push_str(&format!("\tdata_type: {}\n", self.params.data_type));
	}
	
	fn workspace_sz(&self) -> usize {
		self.reduction_workspaces.indices.bytes +
		self.reduction_workspaces.workspace.bytes
	}
}

pub struct ReductionWorkspaces {
	pub desc: ReduceTensorDescriptor,
	pub workspace: gpuMem,
	pub indices: gpuMem
}

impl ReductionWorkspaces {
	// x is the src, y is the dest
	pub fn new(model: &Model, x_desc: &TensorDescriptor, y_desc: &TensorDescriptor) -> Self {
		let desc = ReduceTensorDescriptor::new(
				cudnnReduceTensorOp_t::CUDNN_REDUCE_TENSOR_ADD,
				cudnnDataType_t::CUDNN_DATA_FLOAT, NAN_PROP,
				cudnnReduceTensorIndices_t::CUDNN_REDUCE_TENSOR_NO_INDICES,
				cudnnIndicesType_t::CUDNN_32BIT_INDICES);	
		
		let mut workspace_sz = 0;
		let mut indices_sz = 0;
		
		unsafe {cudnnGetReductionWorkspaceSize(model.handle.cudnn_val, desc.val, x_desc.val, y_desc.val, &mut workspace_sz)}.chk_err();
		unsafe {cudnnGetReductionIndicesSize(model.handle.cudnn_val, desc.val, x_desc.val, y_desc.val, &mut indices_sz)}.chk_err();
		
		let workspace = gpuMem::new(cudnnDataType_t::CUDNN_DATA_INT8, workspace_sz);
		let indices = gpuMem::new(cudnnDataType_t::CUDNN_DATA_INT8, indices_sz);
		// ^ fails (at least in layers/add.rs) w/ CUDNN_STATUS_INVALID_VALUE when workspace_sz & indices_sz are allocated, indicating insufficient workspace sizes
		
		Self {desc, workspace, indices}
	}
}

impl Model {
	pub fn add_sum_reduce(&mut self, params: SumReduceParams) {
		let data_type = params.data_type;
		debug_assert!(self.layers.len() > 0);
		
		let layer_prev_ind = self.layers.len() - 1;
		
		let x = self.layers[layer_prev_ind].y.tensor(); // (input to this layer is the output of the previous layer)
		
		//debug_assert!(x.mem.dataType == data_type);
		
		let output_shape = match &params.sum_type {
			SumType::All => {TensorShape {n: 1, c: 1, h: 1, w: 1}}
			SumType::Axes(axes) => {
				let mut output_shape = x.shape;
				for axis in axes {
					match axis {
						0 => {output_shape.n = 1;}
						1 => {output_shape.c = 1;}
						2 => {output_shape.h = 1;}
						3 => {output_shape.w = 1;}
						_ => {panic!("unknown dimension");}
					}
				}
				output_shape
			}
		};
		
		let y = Tensor::new(data_type, output_shape);
		let reduction_workspaces = ReductionWorkspaces::new(self, &x.desc, &y.desc);
		
		self.layers.push( Layer::new(
			vec![layer_prev_ind],
			InternalTypes::SumReduce(SumReduceInternals {
				params,
				reduction_workspaces
			}),
			y,
			String::from("sum_reduce"),
			data_type
		));
	}
	
	pub fn load_sum_reduce(&mut self, layer_keys: &Vec<KeyPair>) {
		self.add_sum_reduce(SumReduceParams {
			sum_type: find_req_key_parse("sum_type", layer_keys),
			data_type: find_req_key_parse("data_type", layer_keys)
		});
	}
}

