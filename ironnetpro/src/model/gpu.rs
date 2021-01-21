use std::ffi::c_void;
use std::time::{SystemTime, UNIX_EPOCH};
use crate::rand::XorState;
use crate::layers::*;
use crate::saving::{SvStruct, Sv, save_file, load_numpy, ld_gpumem_if_exists,
	config_parse, read_file, find_req_key, find_key_vec, find_req_key_parse};
use crate::string_formatting::print_vec;
use std::env;
use super::{f16, f32_to_f16};

pub struct Model {
	pub handle: Handle,
	
	// dstValue = alpha[0]*resultValue + beta[0]*priorDstValue
	one_f32: Vec<f32>, one_f16: Vec<f16>,
	zero_f32: Vec<f32>, zero_f16: Vec<f16>,
	
	pub eps_f32: Vec<f32>,
	eps_f16: Vec<f16>, // updated from eps_f32 in self.eps()
	
	// logging / debugging (created at model creation, shouldn't be updated after)
	pub t_created: u64, // unix timestamp
	pub model_creation_ironnetpro_version: String, // ironnet version & commit id
	pub model_creation_binary_nm: String, // name of binary
	
	pub batch: u64, // current training batch
	pub batches_per_gradient_step: usize, // batch*batches_per_gradient_step are num seqs used to compute gradient update
	
	// err logging
	pub errs: Vec<f32>,
	pub errs_batch: Vec<u64>, // should be the same length as above
	pub compute_times: Vec<f32>, // should be the same length as above (in secs)
	
	pub test_errs: Vec<f32>, // this and the following line don't need to have the same length as above -- testing can occur less freq than training
	pub test_errs_batch: Vec<u64>,
	
	pub t_batch_start: u64, // used for adding new entry in `compute_times`
	
	////
	
	pub batch_sz: i32, // set when loading, controls how `img` and `time_series` layers are loaded and then initialized
	pub layers: Vec<Layer>,
	
	pub forward_training: bool,
	// ^ when true, cudnnRNNForwardTrainingEx is run
	//   instead of cudnnForwardInferenceEx (the inference
	//   function is presummably faster)
	
	pub shared_workspace: Option<gpuMem>
	// used for mask_future_times_add_softmax_w for the forward pass
	// is not guranteed to have the same information across forward or backward function calls
}

impl Model {
	pub fn new(eps: f32, batch_sz: i32) -> Self {
		let version_string = {
			const VERSION: &str = env!("CARGO_PKG_VERSION");
			const COMMIT_ID: &str = include_str!("../../.git/refs/heads/master");
			const COMMIT_LEN: usize = 9;
			
			const TARGET: &str = env!("TARGET");
			const PROFILE: &str = env!("PROFILE");
			const RUSTV: &str = env!("RUSTV");
			
			let mut commit = String::from(COMMIT_ID);
			commit.truncate(COMMIT_LEN);
			
			format!("{}-{}-{} ({}; {})",
					VERSION, PROFILE, commit, RUSTV, TARGET)
		};
		
		let cur_time = SystemTime::now().duration_since(UNIX_EPOCH)
				.expect("system clock is set before unix epoch").as_secs();
		
		Self {
			handle: Handle::new(),
			
			one_f32: vec![1. as f32],
			one_f16: vec![f32_to_f16(1.)],
			
			zero_f32: vec![0. as f32],
			zero_f16: vec![f32_to_f16(0.)],
			
			eps_f32: vec![eps as f32],
			eps_f16: vec![f32_to_f16(eps)],
			
			t_created: cur_time,
			model_creation_ironnetpro_version: version_string,
			model_creation_binary_nm: env::args().collect::<Vec<String>>()[0].clone(), // name of binary
			
			batch: 0,
			batches_per_gradient_step: 1,
			
			errs: Vec::new(),
			errs_batch: Vec::new(),
			compute_times: Vec::new(),
			
			test_errs: Vec::new(),
			test_errs_batch: Vec::new(),
			
			t_batch_start: cur_time,
			
			batch_sz: batch_sz,
			layers: Vec::new(),
			forward_training: true,
			
			shared_workspace: None
		}
	}
	
	///////////////////////////////////
	// scaling parameters
	pub fn one(&self, data_type: cudnnDataType_t) -> *const c_void {
		match data_type {
			cudnnDataType_t::CUDNN_DATA_FLOAT => {self.one_f32.as_ptr() as *const c_void}
			cudnnDataType_t::CUDNN_DATA_HALF => {self.one_f16.as_ptr() as *const c_void}
			_ => {panic!("datatype not supported");}
		}
	}
	
	pub fn zero(&self, data_type: cudnnDataType_t) -> *const c_void {
		match data_type {
			cudnnDataType_t::CUDNN_DATA_FLOAT => {self.zero_f32.as_ptr() as *const c_void}
			cudnnDataType_t::CUDNN_DATA_HALF => {self.zero_f16.as_ptr() as *const c_void}
			_ => {panic!("datatype not supported");}
		}
	}

	pub fn eps(&self, data_type: cudnnDataType_t) -> *const c_void {
		assert!(f32_to_f16(self.eps_f32[0]) == self.eps_f16[0]);
		
		match data_type {
			cudnnDataType_t::CUDNN_DATA_FLOAT => {self.eps_f32.as_ptr() as *const c_void}
			cudnnDataType_t::CUDNN_DATA_HALF => {self.eps_f16.as_ptr() as *const c_void}
			_ => {panic!("datatype not supported");}
		}
	}
	
	///////////////////////////////
	// remap internal states (beam search)
	//
	// reorder_imgs should be [batch_sz] or less
	// sets states_new[i] = states[reorder_imgs[i]]
	pub fn remap_internal_states(&mut self, reorder_imgs: &Vec<usize>, batch_sz: usize) {
		assert!(reorder_imgs.len() <= batch_sz);
		for layer in self.layers.iter() {
			run_internal!{layer.internals => remap_internal_states(reorder_imgs, batch_sz)};
		}
	}
	
	//////////////////////////
	// forward / backward functions
	impl_get_req_layers!();
	
	pub fn forward_inference(&mut self, layer_ind: usize) {
		self.forward_training = false;
		self.forward(layer_ind);
	}
	
	pub fn forward_training(&mut self, layer_ind: usize) {
		self.forward_training = true;
		self.forward(layer_ind);
	}
	
	// self.forward_training should be set
	pub fn forward(&mut self, layer_ind: usize) {
		let req_layers = self.get_req_layers(layer_ind);
		
		for layer_ind in req_layers.iter() {
			let layer = &self.layers[*layer_ind];
			if layer.run_fwd {continue;}
			
			// Update batch size of layer's output to match that of inputs (i.e., outputs of previous layer)
			// 	This is useful if the number of timepoints to compute is decreased from the initially set tensor size
			//	Exception:  if layer is a multi-head Query, Value or, Key, the shape.n (batch size) represents
			//			the number of heads and not the number of time-points. This isn't a concern for multi-head
			//			layers following the Q, V, K creation because they too will have the first value representing
			//			the same head value.
			match &layer.internals {
				InternalTypes::QKV(_) | InternalTypes::TransposeReshape(_) => {} // skip
				_ => {
					if let Some(&prev_layer_ind) = layer.x_layers.first() {
						// match input type
						match &self.layers[prev_layer_ind].y {
							// input is a sequence, output could either be a sequence or tensor
							Output::RNNData(x) => {
								let seq_len_array = x.seq_len_array.clone();
								let max_seq_len = x.max_seq_len;
								let batch_sz = x.batch_sz;
								
								let layer = &mut self.layers[*layer_ind];
								match &mut layer.y {
									Output::RNNData(y) => {
										y.update_valid_tpoints(&seq_len_array, max_seq_len);
										layer.dy.rnn_data_mut().update_valid_tpoints(&seq_len_array, max_seq_len);
									} Output::Tensor(y) => {
										y.update_batch_sz(max_seq_len*batch_sz);
										layer.dy.tensor_mut().update_batch_sz(max_seq_len*batch_sz);
									}
								}
								
							// input is a tensor, output should only be a tensor
							} Output::Tensor(x) => {
								let batch_sz = x.shape.n;
								
								let layer = &mut self.layers[*layer_ind];
								//println!("layer_ind {} batch_sz {} mem {} shape {}", layer_ind, batch_sz, layer.y.tensor().mem.n_elements, layer.y.tensor().shape.n_elements());
								layer.y.tensor_mut().update_batch_sz(batch_sz);
								layer.dy.tensor_mut().update_batch_sz(batch_sz);
							}
						}
					}
				}
			}
			
			// run internal forward function
			let layer = &self.layers[*layer_ind];
			run_internal!{layer.internals => forward(layer, self) };
			self.layers[*layer_ind].run_fwd = true;
		}
	}
	
	pub fn backward(&mut self, req_layers: &Vec<usize>) {
		for layer_ind in req_layers.iter().rev() {
			//println!("layer_ind {} {}", layer_ind, self.layers[*layer_ind].nm);
			let layer = &self.layers[*layer_ind];
			run_internal!{layer.internals => backward(layer, self) };
		}
	}
	
	// zero out dy only (and ignore dw)
	pub fn zero_out_dy(&mut self) {
		for layer in self.layers.iter_mut() {
			match &mut layer.dy {
				Output::Tensor(tensor) => {tensor.zero_out();}
				Output::RNNData(rnn_data) => {rnn_data.zero_out();}
			}
		}
	}
	
	// zero out dw & dy for all layers
	pub fn zero_out_gradients(&mut self) {
		self.zero_out_dy();
		
		// i.e. the weight gradients
		for layer in self.layers.iter() {
			run_internal!{layer.internals => zero_out_internal_gradients() };
		}
	}
	
	pub fn zero_out_states(&mut self) {
		for layer in self.layers.iter() {
			run_internal!{layer.internals => zero_out_internal_states() };
		}
	}
	
	/////////////////////
	// debug
	pub fn finite_diff_weights_test(&mut self, layer_ind: usize) {
		const EPS: f32 = 1e-1;
		
		self.add_pow_layer_ind(PowParams {alpha: 2., data_type: cudnnDataType_t::CUDNN_DATA_FLOAT}, layer_ind);
		
		let w_tensor_lens = {
			let mut w_tensor_lens = Vec::new();
			let layer = &self.layers[layer_ind];
			run_composite!{layer.internals => ret_weight_lens(gradients(), &mut w_tensor_lens)};
			w_tensor_lens
		};
		
		// loop over tensors (ex if the layer computes y=w*x + b, one iteration would be for w, another for b
		for (w_tensor_ind, w_tensor_len) in w_tensor_lens.iter().enumerate() {
			
			let mut grad_fd = Vec::with_capacity(*w_tensor_len); // finite diff
			let mut grad_analytic = Vec::with_capacity(*w_tensor_len);
			
			// loop over values of w
			for val_ind in 0..*w_tensor_len {
				macro_rules! y_fwd{($sign: expr) => {
					// add/sub EPS to w
					let layer = &self.layers[layer_ind];
					run_composite!{layer.internals => update_weight(gradients(), w_tensor_ind, val_ind, $sign * EPS)};
					
					// forward
					run_internal!{layer.internals => forward(layer, self)};
					
					let layer = &self.layers[self.layers.len() - 1];
					run_internal!{layer.internals => forward(layer, self)};
				};};
				
				let y = |sign| -> f32 {
					y_fwd!(sign);
					
					// return sum y
					self.layers[self.layers.len() - 1].y.ret().iter().sum::<f32>()
				};
				
				let y_p = y( 1.); // y(x + EPS)
				let y_m = y(-2.); // y(x - EPS) (also negating the EPS added in the prev line)
				
				grad_fd.push( (y_p - y_m) / (2.*EPS) );
				
				// backward at y(x)
				grad_analytic.push( {
					// fwd at y(x), adding back previous negatation)
					y_fwd!(1.);
					
					// backward
					{
						self.zero_out_gradients();
						self.layers[self.layers.len() - 1].dy.one_out();
						
						let layer = &self.layers[self.layers.len() - 1];
						run_internal!{layer.internals => backward(layer, self) };
						
						let layer = &self.layers[layer_ind];
						run_internal!{layer.internals => backward(layer, self) };
					}
					
					// analytic gradient for x
					let mut dw = Vec::with_capacity(*w_tensor_len);
					let layer = &self.layers[layer_ind];
					run_composite!{layer.internals => ret_dw(gradients(), w_tensor_ind, &mut dw)};
					dw[val_ind]
				} );
				
				println!("\t{} {}", grad_fd.last().unwrap(), grad_analytic.last().unwrap());
			}
			
			// print diffs
			{
				debug_assert!(grad_fd.len() == grad_analytic.len() && grad_fd.len() == *w_tensor_len);
				
				let norm = |vals: &Vec<f32>| -> f32 { // sqrt(sum(val**2))
					let mut res = 0.;
					for val in vals.iter() {
						res += (*val)*(*val);
					}
					res.sqrt()
				};
				
				// grad_analytic - grad_fd
				let grad_minus = {
					let mut grad_minus = Vec::with_capacity(*w_tensor_len);
					
					for (v_fd, v_analytic) in grad_fd.iter().zip(grad_analytic.iter()) {
						grad_minus.push(*v_analytic - *v_fd);
					}
					grad_minus
				};
				
				let norm_diff = norm(&grad_minus) / (norm(&grad_analytic) + norm(&grad_fd));
				println!("layer {} {}, w_ind {}: {} = {} / ({} + {})   layer_sz: {} {}",
						layer_ind, self.layers[layer_ind].nm,
						w_tensor_ind, norm_diff, norm(&grad_minus),
						norm(&grad_analytic), norm(&grad_fd),
						self.layers[layer_ind].y.tensor().shape.n_elements(),
						self.layers[layer_ind].y.tensor().shape.to_string());
			}
		}
	}
	
	pub fn finite_diff_test(&mut self, layer_ind: usize) {
		const EPS: f32 = 1e-1;
		
		self.add_pow_layer_ind(PowParams {alpha: 2., data_type: cudnnDataType_t::CUDNN_DATA_FLOAT}, layer_ind);
		
		let layer_inputs = self.layers[layer_ind].x_layers.clone();
		
		//println!("{:?}", req_layers);
		
		// layer inputs to change
		for (x_layer_ind_ind, x_layer_ind) in layer_inputs.iter().enumerate() {
			let mut x_vals = self.layers[*x_layer_ind].y.ret();
			
			let mut grad_fd = Vec::with_capacity(x_vals.len()); // finite diff
			let mut grad_analytic = Vec::with_capacity(x_vals.len());
			
			// fill out grad_fd & grad_analytic by looping over x_vals
			for x_ind in 0..x_vals.len() {
				macro_rules! y_fwd{($sign: expr) => {
					// add/sub EPS to x
					x_vals[x_ind] += $sign * EPS;
					self.layers[*x_layer_ind].y.set(&x_vals);
					
					// forward
					let layer = &self.layers[layer_ind];
					run_internal!{layer.internals => forward(layer, self)};
					
					let layer = &self.layers[self.layers.len() - 1];
					run_internal!{layer.internals => forward(layer, self)};
				};};
				
				let mut y = |sign| -> f32 {
					y_fwd!(sign);
					
					// return sum y
					self.layers[self.layers.len() - 1].y.ret().iter().sum::<f32>()
				};
				
				let y_p = y( 1.); // y(x + EPS)
				let y_m = y(-2.); // y(x - EPS) (also negating the EPS added in the prev line)
				
				//println!("y_p {} y_m {}", y_p, y_m);
				
				grad_fd.push( (y_p - y_m) / (2.*EPS) );
				
				// backward at y(x)
				grad_analytic.push( {
					// fwd at y(x), adding back previous negatation)
					y_fwd!(1.);
					
					// backward
					{
						self.zero_out_gradients();
						self.layers[self.layers.len() - 1].dy.one_out();
						
						let layer = &self.layers[self.layers.len() - 1];
						run_internal!{layer.internals => backward(layer, self) };
						
						let layer = &self.layers[layer_ind];
						run_internal!{layer.internals => backward(layer, self) };
						//self.backward(&req_layers);
						//self.backward(&vec![20, 21]);
					}
					
					// analytic gradient for x
					self.layers[*x_layer_ind].dy.ret()[x_ind]
				} );
				
				println!("\t{} {}", grad_fd.last().unwrap(), grad_analytic.last().unwrap());
			} // x value loop
			
			// print diffs
			{
				debug_assert!(grad_fd.len() == grad_analytic.len() && grad_fd.len() == x_vals.len());
				
				let norm = |vals: &Vec<f32>| -> f32 { // sqrt(sum(val**2))
					let mut res = 0.;
					for val in vals.iter() {
						res += (*val)*(*val);
					}
					res.sqrt()
				};
				
				// grad_analytic - grad_fd
				let grad_minus = {
					let mut grad_minus = Vec::with_capacity(x_vals.len());
					
					for (v_fd, v_analytic) in grad_fd.iter().zip(grad_analytic.iter()) {
						grad_minus.push(*v_analytic - *v_fd);
					}
					grad_minus
				};
				
				let norm_diff = norm(&grad_minus) / (norm(&grad_analytic) + norm(&grad_fd));
				println!("layer {} {}, input {} ({}): {} = {} / ({} + {})   layer_sz: {} {}",
						layer_ind, self.layers[layer_ind].nm,
						x_layer_ind_ind, x_layer_ind, norm_diff, norm(&grad_minus),
						norm(&grad_analytic), norm(&grad_fd),
						self.layers[layer_ind].y.tensor().shape.n_elements(),
						self.layers[layer_ind].y.tensor().shape.to_string());
			}
		} // x layer loop
	}

	pub fn finite_diff_full_model_test(&mut self, layer_test_ind: usize, layer_end_ind: usize, rng: &mut XorState) {
		const FD_FACTOR: f32 = 5e0;//5e-1;
		const FD_FACTOR_ADD: f32 = 1e-2;
		
		//if self.layers[layer_test_ind].nm != String::from("FullyConnected") {return;}
		
		// return if no inputs
		let n_inputs = {
			let n_inputs = self.layers[layer_test_ind].x_layers.len();
			if n_inputs == 0 {return;}
			n_inputs
		};
		
		println!("start");
		for _sample in 0..5 {
			// forward
			let req_layers = self.get_req_layers(layer_end_ind);
			{
				self.reset_fwd_cache_flags();
				self.forward(layer_end_ind);
				
				self.zero_out_gradients();
			}
			
			// sum y
			let (layer_output, y_sum) = {
				let layer_output = &self.layers[layer_end_ind];
				let y = &layer_output.y;
				let y_sum = y.ret().iter().sum::<f32>();
				(layer_output, y_sum)
			};
			
			// backward
			{
				layer_output.dy.one_out();
				self.backward(&req_layers);
			}
			
			// choose x to change
			let (x, x_ind, layer_input, layer_input_ind) = {
				let input_sel = if n_inputs == 1 {
					0
				}else{
					rng.usize_range(0, n_inputs)
				};
				
				let layer_output = &self.layers[layer_test_ind];
				let layer_input_ind = layer_output.x_layers[input_sel];
				let layer_input = &self.layers[layer_input_ind];
				let x = &layer_input.y;
				let x_ind = rng.usize_range(0, x.n_elements());
				(x, x_ind, layer_input, layer_input_ind)
			};
			
			// get analytic gradient for x
			let analytic_grad = layer_input.dy.ret()[x_ind];
			
			// add eps to x
			let fd_eps = {
				let mut x_vals = x.ret();
				let fd_eps = x_vals[x_ind]*FD_FACTOR;
				x_vals[x_ind] += fd_eps + FD_FACTOR_ADD;
				x.set(&x_vals);
				fd_eps
			};
			
			// forward then get summed y w/ changed x
			let y_eps_sum = {
				self.reset_fwd_cache_flags();
				self.layers[layer_input_ind].run_fwd = true;
				self.forward(layer_end_ind);
				self.layers[layer_end_ind].y.ret().iter().sum::<f32>()
			};
			
			// finite diff gradient
			let fd_grad = (y_eps_sum - y_sum) / (fd_eps + FD_FACTOR_ADD);
			
			// ratio & printing
			let ratio = if fd_grad == analytic_grad {1.} else {analytic_grad / fd_grad};
			//if y_eps_sum == y_sum {continue;}
			println!("{} {} {} y {} y_eps {} {}",
					ratio, fd_grad, analytic_grad,
					y_sum, y_eps_sum, self.layers[layer_test_ind].nm);
		}
	}
	
	//////////////////
	// saving / loading
	pub fn sv(&self, model_dir: &str) {
		let mut config_txt = String::new();
		
		{ // model meta parameters
			config_txt.push_str("{\n");
			config_txt.push_str(&format!("\tEPS: {}\n", self.eps_f32[0]));
			config_txt.push_str(&format!("\tt_created: {}\n", self.t_created));
			config_txt.push_str(&format!("\tmodel_creation_ironnetpro_version: {}\n", self.model_creation_ironnetpro_version));
			config_txt.push_str(&format!("\tmodel_creation_binary_nm: {}\n", self.model_creation_binary_nm));
			config_txt.push_str(&format!("\tbatch: {}\n", self.batch));
			config_txt.push_str(&format!("\tbatches_per_gradient_step: {}\n", self.batches_per_gradient_step));
			config_txt.push_str("}\n");
		}
		
		self.errs.sv(model_dir, "errs");
		self.errs_batch.sv(model_dir, "errs_batch");
		self.compute_times.sv(model_dir, "compute_times");
		
		self.test_errs.sv(model_dir, "test_errs");
		self.test_errs_batch.sv(model_dir, "test_errs_batch");
		
		// layers
		for (layer_ind, layer) in self.layers.iter().enumerate() {
			{ // layer meta data / arch data 
				config_txt.push_str("\n{\n");
				config_txt.push_str(&format!("\ttype: {}\n", layer.nm));
				
				// layer inputs
				if layer.x_layers.len() > 0 {
					config_txt.push_str("\tx_layer_inputs: ");
					print_vec(&layer.x_layers, &mut config_txt);
				}
				
				run_internal!{layer.internals => sv_arch(&mut config_txt)};
				
				config_txt.push_str("}\n");
			}
			
			{ // weights / array data (ex. optimizer data)
				let file_nm = &format!("{}_{}", layer_ind, layer.nm);		
				run_internal!{layer.internals => sv_weights(layer, self, &format!("{}/weights", model_dir), file_nm) };
				
				///////////////
				// optimizer variables
				
				if let Some(weights_rms_tmp) = &layer.weights_rms_tmp {
					weights_rms_tmp.sv(&format!("{}/weights_rms_tmp", model_dir), file_nm);
				}
				
				if let Some(weights_adam) = &layer.weights_adam {
					weights_adam.m.sv(&format!("{}/weights_adam_m", model_dir), file_nm);
					weights_adam.v.sv(&format!("{}/weights_adam_v", model_dir), file_nm);
				}
			}
		}
		
		// convert string to bytes
		let mut res = Vec::with_capacity(config_txt.len());
		config_txt.sv_buf(&mut res);
		
		save_file(model_dir, LAYER_CONFIG_NM, &res);
		//println!("{}", config_txt);
	}
	
	pub fn sv_gradients_and_outputs(&self, model_dir: &str) {
		for (layer_ind, layer) in self.layers.iter().enumerate() {
			let file_nm = &format!("{}_{}", layer_ind, layer.nm);
			
			run_internal!{layer.internals => sv_gradients(&format!("{}/gradients", model_dir), file_nm) };
			
			layer.y.sv(&format!("{}/outputs", model_dir), file_nm);
			layer.dy.sv(&format!("{}/d_outputs", model_dir), file_nm);
		}
	}
	
	pub fn load(model_dir: &str, eps: f32, batch_sz: i32) -> Model {
		let mut rng = XorState::clock_init();
		
		let key_sets = config_parse(read_file(&format!("{}/{}", model_dir, LAYER_CONFIG_NM)));
		
		let mut model = {
			let model_keys = &key_sets[0];
			let mut model = Model::new(eps, batch_sz);
			
			model.t_created = find_req_key_parse("t_created", model_keys);
			model.model_creation_ironnetpro_version = find_req_key_parse("model_creation_ironnetpro_version", model_keys);
			model.model_creation_binary_nm = find_req_key_parse("model_creation_binary_nm", model_keys);
			
			model.batch = find_req_key_parse("batch", model_keys);
			model.batches_per_gradient_step = find_req_key_parse("batches_per_gradient_step", model_keys);
			
			model.errs = load_numpy(&format!("{}/errs.npy", model_dir));
			model.errs_batch = load_numpy(&format!("{}/errs_batch.npy", model_dir));
			model.compute_times = load_numpy(&format!("{}/compute_times.npy", model_dir));
			
			model.test_errs = load_numpy(&format!("{}/test_errs.npy", model_dir));
			model.test_errs_batch = load_numpy(&format!("{}/test_errs_batch.npy", model_dir));
			
			model
		};
		
		for layer_keys in key_sets.iter().skip(1) { // skip model meta params
			let layer_type = find_req_key("type", &layer_keys);
			let x_layers: Vec<usize> = find_key_vec("x_layer_inputs", layer_keys);
			
			// check that inputs do not exceed already added layers
			for x_layer in x_layers.iter() {
				assert!(*x_layer < model.layers.len(),
						"layer ind input {} exceeds layers added: {}",
						*x_layer, model.layers.len());
			}
			
			check_layer_input_dims(&x_layers, &layer_type);
			
			// initialize layer
			match layer_type.as_str() {
				"conv" => {model.load_conv(layer_keys, &mut rng);}
				"pooling" => {model.load_max_pooling(layer_keys);}
				"softmax" => {model.load_softmax(layer_keys);}
				"softmax_across_w" => {model.load_softmax_across_w(layer_keys);}
				"softmax_log" => {model.load_softmax_log(layer_keys);}
				"relu" => {model.load_relu(layer_keys);}
				"tanh" => {model.load_tanh(layer_keys);}
				"add" => {model.load_add(&x_layers, layer_keys);}
				"mul" => {model.load_mul(&x_layers, layer_keys);}
				"sum_reduce" => {model.load_sum_reduce(layer_keys);}
				"pow" => {model.load_pow(&x_layers, layer_keys);}
				"LSTM" => {model.load_lstm(layer_keys, &mut rng);}
				"imgs" => {model.load_imgs(layer_keys);}
				"FullyConnected" => {model.load_fully_connected(layer_keys, &mut rng);}
				"FullyConnectedWBias" => {model.load_fully_connected_w_bias(layer_keys, &mut rng);}
				"FullyConnectedWBiasRelu" => {model.load_fully_connected_w_bias_relu(layer_keys, &mut rng);}
				"FullyConnectedWBiasReluInputSupplied" => {model.load_fully_connected_w_bias_relu_input_supplied(&x_layers, layer_keys, &mut rng);}
				"QKV" => {model.load_QKV_layer(&x_layers, layer_keys, &mut rng);}
				"MulQK" => {model.load_mul_Q_K_layer(&x_layers, layer_keys);}
				"Scale" => {model.load_scale(layer_keys);}
				"MaskFutureTimes" => {model.load_mask_future_times(layer_keys);}
				"MulSoftmaxQKAndV" => {model.load_mul_softmaxQK_and_V(&x_layers, layer_keys);}
				"MulQAndPos" => {model.load_mul_Q_and_pos(&x_layers, layer_keys, &mut rng);}
				"QKPlusQPosMaskFutureTimesSoftmaxW" => {model.load_QK_plus_Qpos_mask_future_times_softmax_w(&x_layers, layer_keys, &mut rng);}
				"QKPlusQPosMaskFutureTimesSoftmaxWMulV" => {model.load_QK_plus_Qpos_mask_future_times_softmaxw_mul_V(&x_layers, layer_keys, &mut rng);}
				"TransposeReshape" => {model.load_transpose_reshape(layer_keys);}
				"Bias" => {model.load_bias(layer_keys, &mut rng);}
				"BiasChannels" => {model.load_bias_channels(layer_keys, &mut rng);}
				"time_series" => {model.load_time_series(layer_keys);}
				"elementwise_affine" => {model.load_elementwise_affine(layer_keys);}
				_ => {panic!("unknown layer type: {}", layer_type);}
			}
			
			debug_assert!(x_layers == model.layers.last().unwrap().x_layers);
			
			// load weights / array data (ex optimizer data)
			{
				let layer_ind = model.layers.len() - 1;
				let layer = &model.layers.last().unwrap();
				let file_nm = &format!("{}_{}", layer_ind, layer.nm);
				
				run_internal!{layer.internals =>
							ld_weights(layer, &model, &format!("{}/weights", model_dir), file_nm) };
				
				// optimizer data
				let layer = model.layers.last_mut().unwrap();
				layer.weights_rms_tmp = ld_gpumem_if_exists(&format!("{}/weights_rms_tmp", model_dir), file_nm);
				
				if let Some(weights_adam_m) = ld_gpumem_if_exists(&format!("{}/weights_adam_m", model_dir), file_nm) {
					println!("loaded adam weights");
					layer.weights_adam = Some(WeightsAdam {
							m: weights_adam_m,
							v: ld_gpumem_if_exists(&format!("{}/weights_adam_v", model_dir), file_nm).unwrap()
					});
				}
			}
		}
		model
	}
	
	pub fn input_layer_inds(&self) -> Vec<usize> {
		let mut inds = Vec::new();
		for (ind, layer) in self.layers.iter().enumerate() {
			match layer.internals {
				InternalTypes::Conv(_) |
				InternalTypes::Pooling(_) |
				InternalTypes::Softmax(_) |
				InternalTypes::Activation(_) |
				InternalTypes::Add(_) |
				InternalTypes::Mul(_) |
				InternalTypes::SumReduce(_) |
				InternalTypes::Pow(_) |
				InternalTypes::LSTM(_) |
				InternalTypes::QKV(_) |
				InternalTypes::QKPlusQPosMaskFutureTimesSoftmaxW(_) |
				InternalTypes::QKPlusQPosMaskFutureTimesSoftmaxWMulV(_) |
				
				InternalTypes::MulQK(_) |
				InternalTypes::MulSoftmaxQKAndV(_) |
				InternalTypes::MaskFutureTimes(_) |
				
				InternalTypes::MaskFutureTimesAdd(_) |
				InternalTypes::MaskFutureTimesAddSoftmaxW(_) |
				
				InternalTypes::Scale(_) |
				InternalTypes::TransposeReshape(_) |
				InternalTypes::Bias(_) |
				InternalTypes::BiasChannelsCustom(_) |
				InternalTypes::FullyConnected(_) |
				InternalTypes::FullyConnectedWBias(_) |
				InternalTypes::FullyConnectedWBiasRelu(_) |
				InternalTypes::MulQAndPos(_) |
				InternalTypes::ElementwiseAffine(_) => {}
				
				InternalTypes::Img(_) |
				InternalTypes::TimeSeries(_) => {
					inds.push(ind);
				}
			}
		}
		inds
	}
	
	pub fn log_err(&mut self) {
		let err = self.layers.last().unwrap().y.ret();
		debug_assert!(err.len() == 1);
		self.errs.push(err[0]);
		self.errs_batch.push(self.batch);
	}
	
	// if a testing or validation set is used
	pub fn log_test_err(&mut self, err: f32) {
		self.test_errs.push(err);
		self.test_errs_batch.push(self.batch);
	}

	/////////////////
	
	// allocate shared workspace
	//	is not guranteed to have same contents between forward/backward calls
	pub fn allocate_shared_workspace(&mut self, data_type: cudnnDataType_t, n_elements: usize) {
		// workspace
		// 	often used w/ forward because cudnnOpTensor cannot work in-place w/ same input & output buffer
		if let Some(shared_workspace) = &mut self.shared_workspace {
			// reallocate to larger size
			if shared_workspace.bytes < data_type.bytes()*n_elements {
				*shared_workspace = gpuMem::new(data_type, n_elements);
			}
		}else{
			self.shared_workspace = Some(gpuMem::new(data_type, n_elements));
		}
	}
}

