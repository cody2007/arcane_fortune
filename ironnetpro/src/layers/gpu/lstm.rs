use std::os::raw::{c_int, c_ulonglong};
use std::ffi::c_void;
use std::ptr::null_mut;
use super::*;
use crate::saving::{KeyPair, find_req_key_parse, sv_w_shape, load_numpy};
use crate::layers::LSTMParams;

pub struct LSTMInternals {
	params: LSTMParams,
	
	rnn_desc: RNNDescriptor,
	workspace: gpuMem,
	reserve: gpuMem,
	
	filter: Filter3,
	dfilter: Filter3,
	
	// internal LSTM memory cells
	h: Tensor,
	c: Tensor,
}

const LSTM_ALG: cudnnRNNAlgo_t = cudnnRNNAlgo_t::CUDNN_RNN_ALGO_STANDARD;
const LSTM_MODE: cudnnRNNMode_t = cudnnRNNMode_t::CUDNN_LSTM;
const LSTM_DIRECTION: cudnnDirectionMode_t = cudnnDirectionMode_t::CUDNN_UNIDIRECTIONAL;
const LSTM_INPUT: cudnnRNNInputMode_t = cudnnRNNInputMode_t::CUDNN_LINEAR_INPUT;

/////////////////// for: sv_weights() and ld_weights()
// layer_id:
// "Values of 0, 1, 2 and 3 reference bias applied to the input from the previous layer"
// "Values of 4, 5, 6 and 7 reference bias applied to the recurrent input"

// "Values 0 and 4 reference the input gate."
// "Values 1 and 5 reference the forget gate."
// "Values 2 and 6 reference the new memory gate."
// "Values 3 and 7 reference the output gate."

// --cudnnAPI DA-09702-001_v7.6.5 | page 227
const LSTM_LAYER_NMS: &[&str] = &["Wi", "Wf", "Wc", "Wo", "Ri", "Rf", "Rc", "Ro"];

impl Run for LSTMInternals {
	fn forward(&self, layer: &Layer, model: &Model) {
		let x = model.layers[layer.x_layers[0]].y.rnn_data(); // output of input layer is the input for this layer
		let y = layer.y.rnn_data();
		
		if model.forward_training {
			unsafe {cudnnRNNForwardTrainingEx(model.handle.cudnn_val,
					self.rnn_desc.val,
					
					x.desc.val, x.mem.val,
					
					self.h.desc.val, null_mut(), // hx (initial state)
					self.c.desc.val, null_mut(), // cx (initial state)
					
					self.filter.desc.val, self.filter.mem.val,
					
					//////// outputs
					y.desc.val, y.mem.val,
					
					self.h.desc.val, null_mut(), // hy (final state)
					self.c.desc.val, null_mut(), // cy (final state)
					null_mut(), null_mut(),
					null_mut(), null_mut(),
					null_mut(), null_mut(),
					null_mut(), null_mut(),
					
					self.workspace.val, self.workspace.bytes,
					self.reserve.val, self.reserve.bytes
			)}.chk_err();
		
		/////// inference
		}else{
			unsafe {cudnnRNNForwardInferenceEx(model.handle.cudnn_val,
					self.rnn_desc.val,
					
					x.inference_desc.val, x.mem.val,
					
					self.h.desc.val, self.h.mem.val, // hx (initial state)
					self.c.desc.val, self.c.mem.val, // cx (initial state)
					
					self.filter.desc.val, self.filter.mem.val,
					
					//////// outputs
					y.inference_desc.val, y.mem.val,
					
					self.h.desc.val, self.h.mem.val, // hy (final state)
					self.c.desc.val, self.c.mem.val, // cy (final state)
					null_mut(), null_mut(),
					null_mut(), null_mut(),
					null_mut(), null_mut(),
					null_mut(), null_mut(),
					
					self.workspace.val, self.workspace.bytes
			)}.chk_err();
		}
	}
	
	fn backward(&self, layer: &Layer, model: &Model) {
		let layer_prev = &model.layers[layer.x_layers[0]];
		let x = layer_prev.y.rnn_data();
		let dx = layer_prev.dy.rnn_data();
		let y = layer.y.rnn_data();
		let dy = layer.dy.rnn_data();
		
		// backward data should be called before backward weights
		
		unsafe {cudnnRNNBackwardDataEx(model.handle.cudnn_val,
				self.rnn_desc.val,
				y.desc.val, y.mem.val,
				dy.desc.val, dy.mem.val,
				
				null_mut(), null_mut(), // dcDesc, dcAttn
				self.h.desc.val, null_mut(), // dhy (deriv wrt final state)
				self.c.desc.val, null_mut(), // dcy
				
				self.filter.desc.val, self.filter.mem.val,
				
				// initial state
				self.h.desc.val, null_mut(),
				self.c.desc.val, null_mut(),
				
				///////////////// outputs
				dx.desc.val, dx.mem.val,
				
				// derivative wrt to initial internal states
				self.h.desc.val, null_mut(), // dhx
				self.c.desc.val, null_mut(), // dcx
				null_mut(), null_mut(), // dkDesc, dkeys
				
				self.workspace.val, self.workspace.bytes,
				self.reserve.val, self.reserve.bytes)}.chk_err();
		
		unsafe {cudnnRNNBackwardWeightsEx(model.handle.cudnn_val,
				self.rnn_desc.val,
				
				x.desc.val, x.mem.val,
				
				self.h.desc.val, null_mut(), // hx (initial state)
				y.desc.val, y.mem.val,
				
				self.workspace.val, self.workspace.bytes,
				self.dfilter.desc.val, self.dfilter.mem.val,
				self.reserve.val, self.reserve.bytes)}.chk_err();
	}
	
	fn zero_out_internal_states(&self) {
		self.h.zero_out();
		self.c.zero_out();
	}
	
	fn zero_out_internal_gradients(&self) {
		self.dfilter.zero_out();
	}
	
	fn ret_internal_states(&self) -> Option<Vec<Vec<f32>>> {
		Some(vec![ self.h.ret(), self.c.ret() ])
	}
	
	fn set_internal_states(&self, states: &Vec<Vec<f32>>) {
		self.h.set(&states[0]);
		self.c.set(&states[1]);
	}
	
	// reorder batch dimension (useful with beam search)
	fn remap_internal_states(&self, reorder_imgs: &Vec<usize>, batch_sz: usize) {
		assert!(reorder_imgs.len() <= batch_sz);
		let states = self.ret_internal_states().unwrap();
		let mut states_new = Vec::with_capacity(states.len());
		
		let n_layers = self.params.n_layers as usize;
		let hidden_sz = self.params.hidden_sz as usize;
		
		// for each type of state, ex. `h` and `c`
		for state in states.iter() {
			// state: [n_layers, batch_sz, hidden_sz]
			assert!(state.len() == (n_layers * batch_sz * hidden_sz));
			
			let mut state_new = vec!{0.; state.len()}; // [n_layers, batch_sz, dim_sz]
			
			for pseudo_layer in 0..n_layers {
				let layer_off = pseudo_layer*batch_sz*hidden_sz;
				
				// reorder imgs (across batch_sz middle dim)
				for (reorder_img, state_new_img) in reorder_imgs.iter()
							.zip(state_new.chunks_mut(hidden_sz)
								        .skip(pseudo_layer*batch_sz)) {
					
					state_new_img.clone_from_slice(
							&state[layer_off +  reorder_img    * hidden_sz ..
								 layer_off + (reorder_img+1) * hidden_sz]);
				}
			}
			
			states_new.push(state_new);
		}
		
		self.set_internal_states(&states_new);
	}
	
	fn gradients(&self) -> Vec<Weights> {
		vec![Weights {
			w_desc: self.filter.tensor_desc.val, 
			w_mem: self.filter.mem.val,
			dw_desc: self.dfilter.tensor_desc.val,
			dw_mem: self.dfilter.mem.val,
			len: self.filter.mem.n_elements,
			data_type: self.filter.mem.dataType
		}]
	}
	
	fn sv_arch(&self, txt: &mut String) {
		txt.push_str(&format!("\thidden_sz: {}\n", self.params.hidden_sz));
		txt.push_str(&format!("\tn_layers: {}\n", self.params.n_layers));
		txt.push_str(&format!("\tdropout_chance: {}\n", self.params.dropout_chance));
		txt.push_str(&format!("\tnorm_scale: {}\n", self.params.norm_scale));
		txt.push_str(&format!("\tdata_type: {}\n", self.params.data_type));
	}
	
	fn sv_weights(&self, layer: &Layer, model: &Model, save_dir: &str, file_nm: &str) {
		let x = model.layers[layer.x_layers[0]].y.rnn_data(); // (input to this layer is the output of the previous layer)
		
		let mat_desc = FilterDescriptor::create();
		let mut mat_gpu: gpuMem_t = null_mut(); // we do not use gpuMem here because
									  // cudnn will set it to a valid pointer to
									  // already-allocated memory.
							  		  // (gpuMem will free memory on dropping)
		
		for pseudo_layer in 0..self.params.n_layers { // hidden layer number
			for (layer_id, layer_nm) in LSTM_LAYER_NMS.iter().enumerate() {
				macro_rules! sv_vals{($prefix_nm: expr) => {
					let mat_shape = mat_desc.shape3();
					let mat = ret_raw::<f32>(&mat_gpu, mat_shape.n_elements());
					sv_w_shape(&mat, save_dir, &format!("{}_{}_{}{}", file_nm, pseudo_layer, $prefix_nm, layer_nm), Some(mat_shape));
				};};
				
				// save weights
				{
					unsafe {cudnnGetRNNLinLayerMatrixParams(model.handle.cudnn_val,
							self.rnn_desc.val, pseudo_layer,
							x.timestep_tensor_desc_vals[0],
							self.filter.desc.val, self.filter.mem.val,
							layer_id as c_int, mat_desc.val, &mut mat_gpu)}.chk_err();
					sv_vals!("");
				}
				
				// save bias
				{
					unsafe {cudnnGetRNNLinLayerBiasParams(model.handle.cudnn_val,
							self.rnn_desc.val, pseudo_layer,
							x.timestep_tensor_desc_vals[0],
							self.filter.desc.val, self.filter.mem.val,
							layer_id as c_int, mat_desc.val, &mut mat_gpu)}.chk_err();
					sv_vals!("b");
				}
			}
		}
	}
	
	fn sv_gradients(&self, save_dir: &str, file_nm: &str) {
		self.dfilter.sv(save_dir, file_nm);
	}
	
	fn ld_weights(&self, layer: &Layer, model: &Model, save_dir: &str, file_nm: &str) {
		let x = model.layers[layer.x_layers[0]].y.rnn_data(); // (input to this layer is the output of the previous layer)
		
		let mat_desc = FilterDescriptor::create();
		let mut mat_gpu: gpuMem_t = null_mut(); // we do not use gpuMem here because
									  // cudnn will set it to a valid pointer to
									  // already-allocated memory.
							  		  // (gpuMem will free memory on dropping)
		
		for pseudo_layer in 0..self.params.n_layers { // hidden layer number
			for (layer_id, layer_nm) in LSTM_LAYER_NMS.iter().enumerate() {
				macro_rules! ld_vals{($prefix_nm: expr) => {
					let mat_shape = mat_desc.shape3();
					let file_nm = format!("{}/{}_{}_{}{}.npy", save_dir, file_nm, pseudo_layer, $prefix_nm, layer_nm);
					
					let vals = load_numpy::<f32>(&file_nm);
					
					// set gpu mem
					assert!(vals.len() == mat_shape.n_elements());
					assert!(self.filter.mem.dataType == cudnnDataType_t::CUDNN_DATA_FLOAT);
					assert!(vals.len() <= self.filter.mem.n_elements, 
							"loaded length: {}, expected: {}", vals.len(), self.filter.mem.n_elements);
					
					unsafe {cudaMemcpy(
							mat_gpu as *mut c_void,
							vals.as_ptr() as *const c_void,
							vals.len() * self.filter.mem.dataType.bytes(),
							cudaMemcpyKind::cudaMemcpyHostToDevice
					)}.chk_err();
				};};
				
				// save weights
				{
					unsafe {cudnnGetRNNLinLayerMatrixParams(model.handle.cudnn_val,
							self.rnn_desc.val, pseudo_layer,
							x.timestep_tensor_desc_vals[0],
							self.filter.desc.val, self.filter.mem.val,
							layer_id as c_int, mat_desc.val, &mut mat_gpu)}.chk_err();
					ld_vals!("");
				}
				
				// save bias
				{
					unsafe {cudnnGetRNNLinLayerBiasParams(model.handle.cudnn_val,
							self.rnn_desc.val, pseudo_layer,
							x.timestep_tensor_desc_vals[0],
							self.filter.desc.val, self.filter.mem.val,
							layer_id as c_int, mat_desc.val, &mut mat_gpu)}.chk_err();
					ld_vals!("b");
				}
			}
		}
	}
	
	fn workspace_sz(&self) -> usize {
		self.workspace.bytes +
		self.reserve.bytes + 
		self.h.mem.bytes +
		self.c.mem.bytes
	}
}

impl Model {
	pub fn add_lstm(&mut self, params: LSTMParams, rng: &mut XorState) {
		let data_type = params.data_type;
		debug_assert!(self.layers.len() > 0);
		
		let layer_prev_ind = self.layers.len() - 1;
		let layer_prev = &self.layers[layer_prev_ind];
			
		let x = layer_prev.y.rnn_data(); // (input to this layer is the output of the previous layer)
		let y = RNNData::new(data_type, x.max_seq_len, x.batch_sz, params.hidden_sz, &x.seq_len_array);
		
		let dropout_desc = DropoutDescriptor::new(&self.handle, params.dropout_chance, rng.gen() as c_ulonglong);
		
		let rnn_desc = RNNDescriptor::new(&self.handle, params.hidden_sz, params.n_layers,
				dropout_desc, LSTM_INPUT, LSTM_DIRECTION,
				LSTM_MODE, LSTM_ALG, data_type);
		
		unsafe {cudnnSetRNNPaddingMode(rnn_desc.val, cudnnRNNPaddingMode_t::CUDNN_RNN_PADDED_IO_ENABLED)}.chk_err();
		
		// workspaces
		let (workspace, reserve) = {
			////// workspace
			let mut workspace_sz: size_t = 0;
			unsafe {cudnnGetRNNWorkspaceSize(self.handle.cudnn_val, rnn_desc.val, x.max_seq_len,
					x.timestep_tensor_desc_vals.as_ptr() as *const cudnnTensorDescriptor_t, &mut workspace_sz)}.chk_err();
			
			let workspace = gpuMem::new_bytes(workspace_sz);
			
			/////// reserve
			let mut reserve_sz: size_t = 0;
			unsafe {cudnnGetRNNTrainingReserveSize(self.handle.cudnn_val, rnn_desc.val, x.max_seq_len,
					y.timestep_tensor_desc_vals.as_ptr() as *const cudnnTensorDescriptor_t, &mut reserve_sz)}.chk_err();
			
			let reserve = gpuMem::new_bytes(reserve_sz);
			(workspace, reserve)
		};
		
		// filter
		let (filter, dfilter) = {
			let mut param_sz = 0;
			unsafe {cudnnGetRNNParamsSize(self.handle.cudnn_val, rnn_desc.val,
				x.timestep_tensor_desc_vals[0], &mut param_sz, data_type)}.chk_err();
			
			param_sz /= data_type.bytes();
			
			(Filter3::new_norm(data_type, param_sz, params.norm_scale, rng),
			 Filter3::zeros(data_type, param_sz))
		};
		
		///////// internal state tensor descriptor
		let hx_shape = Tensor3Shape {
				dim1: params.n_layers,
				dim2: x.timestep_shape.dim1, // i.e., the batch size
				dim3: params.hidden_sz 
		};
		
		self.layers.push( Layer::new_time_series(
			vec!{layer_prev_ind; 1},
			InternalTypes::LSTM(LSTMInternals {
					params,
					
					rnn_desc,
					workspace,
					reserve,
					
					filter,
					dfilter,
					
					h: Tensor::zeros3(data_type, &hx_shape),
					c: Tensor::zeros3(data_type, &hx_shape),
			}),
			y,
			String::from("LSTM"),
			data_type
		));
	}
	
	pub fn load_lstm(&mut self, layer_keys: &Vec<KeyPair>, rng: &mut XorState) {
		self.add_lstm(LSTMParams {
			hidden_sz: find_req_key_parse("hidden_sz", layer_keys),
			n_layers: find_req_key_parse("n_layers", layer_keys),
			dropout_chance: find_req_key_parse("dropout_chance", layer_keys),
			norm_scale: find_req_key_parse("norm_scale", layer_keys),
			data_type: find_req_key_parse("data_type", layer_keys)
		}, rng);
	}
}

