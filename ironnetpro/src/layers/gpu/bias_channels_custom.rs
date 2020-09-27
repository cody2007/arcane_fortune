use super::*;
use crate::layers::BiasParams;
use crate::saving::{KeyPair, find_req_key_parse};

pub struct BiasChannelsCustomInternals {
	params: BiasParams,
	op_tensor_desc: OpTensorDescriptor,
	bias: Filter,
	dbias: Filter,
}

impl Run for BiasChannelsCustomInternals {
	fn forward(&self, layer: &Layer, model: &Model) {
		#[cfg(feature="profile")]
		let _g = Guard::new("bias channels custom fwd");
		
		debug_assert!(layer.x_layers.len() == 1);
		let x = model.layers[layer.x_layers[0]].y.tensor(); // output of input layer is the input for this layer
		let y = layer.y.tensor();
		
		unsafe {cudnnOpTensor(model.handle.cudnn_val,
				self.op_tensor_desc.val,
				
				model.one(layer.data_type),
				x.desc.val, x.mem.val,
				
				model.one(layer.data_type),
				self.bias.tensor_desc.val, self.bias.mem.val,
				
				model.zero(layer.data_type),
				y.desc.val, y.mem.val)}.chk_err();
	}
	
	fn backward(&self, layer: &Layer, model: &Model) {
		#[cfg(feature="profile")]
		let _g = Guard::new("bias channels custom bwd");
		
		debug_assert!(layer.x_layers.len() == 1);
		let in_layer = &model.layers[layer.x_layers[0]]; // output of input layer is the input for this layer
		
		let dx = &in_layer.dy.tensor(); 
		let dy = &layer.dy.tensor();
		
		////////// dx
		unsafe {cudnnAddTensor(model.handle.cudnn_val,
					model.one(layer.data_type),
					dy.desc.val, dy.mem.val,
					
					model.one(layer.data_type),
					dx.desc.val, dx.mem.val)}.chk_err();
		
		/////// dbias
		{
			/*unsafe {cudnnReduceTensor(
					dy.mem.val,
					self.dbias.mem.val)}.chk_err();*/
			
			let n_batches = (dy.shape.n * dy.shape.c) as usize;
			let bias_sz = self.dbias.shape.n_elements();
			
			assert!((n_batches*bias_sz) == dy.shape.n_elements());
			
			unsafe {dbias_plus_dy(self.dbias.mem.val, dy.mem.val, n_batches, bias_sz)};
		}
	}
	
	fn zero_out_internal_gradients(&self) {
		self.dbias.zero_out();
	}
	
	fn gradients(&self) -> Vec<Weights> {
		vec![Weights {
			w_desc: self.bias.tensor_desc.val, 
			w_mem: self.bias.mem.val,
			dw_desc: self.dbias.tensor_desc.val,
			dw_mem: self.dbias.mem.val,
			len: self.bias.mem.n_elements,
			data_type: self.bias.mem.dataType
		}]
	}
	
	fn sv_arch(&self, txt: &mut String) {
		txt.push_str(&format!("\tnorm_scale: {}\n", self.params.norm_scale));
		txt.push_str(&format!("\tdata_type: {}\n", self.params.data_type));
	}

	fn sv_weights(&self, _: &Layer, _: &Model, save_dir: &str, file_nm: &str) {
		self.bias.sv(save_dir, file_nm);
	}
	
	fn sv_gradients(&self, save_dir: &str, file_nm: &str) {
		self.dbias.sv(save_dir, file_nm);
	}
	
	fn ld_weights(&self, _: &Layer, _: &Model, save_dir: &str, file_nm: &str) {
		self.bias.ld(save_dir, file_nm);
	}
}

impl Model {
	// bias is applied across all images & channels
	pub fn add_bias_channels_custom(&mut self, params: BiasParams, rng: &mut XorState) {
		let data_type = params.data_type;
		debug_assert!(self.layers.len() > 0);
		debug_assert!(params.data_type == cudnnDataType_t::CUDNN_DATA_FLOAT);
		
		let layer_prev = self.layers.len() - 1;
		
		let x_shape = self.layers[layer_prev].y.ravel_time_shape(); // (input to this layer is the output of the previous layer)
		
		let bias_shape = FilterShape {
				k: 1,
				c: 1,
				h: x_shape.h,
				w: x_shape.w
		};
		
		let op_tensor_desc = OpTensorDescriptor::new(cudnnOpTensorOp_t::CUDNN_OP_TENSOR_ADD, data_type, NAN_PROP);
		
		let bias = Filter::new_norm(data_type, bias_shape, params.norm_scale, rng);
		let dbias = Filter::zeros(data_type, bias_shape);
		
		let y = Tensor::new(data_type, x_shape);
		let dy = Tensor::new(data_type, x_shape);
		
		self.layers.push( Layer::new_w_dy(
			vec![layer_prev],
			InternalTypes::BiasChannelsCustom(BiasChannelsCustomInternals {
				params,
				op_tensor_desc,
				bias,
				dbias
			}),
			y, dy,
			String::from("BiasChannelsCustom"),
			data_type
		));
	}

	pub fn load_bias_channels_custom(&mut self, layer_keys: &Vec<KeyPair>, rng: &mut XorState) {
		self.add_bias_channels_custom(BiasParams {
			norm_scale: find_req_key_parse("norm_scale", layer_keys),
			data_type: find_req_key_parse("data_type", layer_keys)
		}, rng);
	}
}

