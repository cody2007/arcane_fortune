use std::os::raw::{c_int};
use crate::rand::XorState;
use crate::layers::*;
use crate::saving::{config_parse, read_file, find_req_key, find_key_vec};

pub struct ModelCPU {
	pub layers: Vec<LayerCPU>,
	pub layer_internals: Vec<InternalTypesCPU>,
	pub batch_sz: c_int
}

impl ModelCPU {
	pub fn new(batch_sz: c_int) -> Self {
		Self {
			layers: Vec::new(),
		  	layer_internals: Vec::new(),
		  	batch_sz
		}
	}
	
	//////////////////////////// forward
	impl_get_req_layers!();
	
	pub fn forward(&mut self, layer_ind: usize) {
		let req_layers = self.get_req_layers(layer_ind);
		
		for layer_ind in req_layers.iter() {
			let layer = &self.layers[*layer_ind];
			if layer.run_fwd {continue;}
			
			// update timepoints and batch size
			if let Some(&prev_layer_ind) = layer.x_layers.first() {
				// match input type
				match &self.layers[prev_layer_ind].y {
					// input is a sequence, output could either be a sequence or tensor
					OutputCPU::RNNData(x) => {
						let seq_len_array = x.seq_len_array.clone();
						let max_seq_len = x.max_seq_len;
						let batch_sz = x.batch_sz;
						
						let layer = &mut self.layers[*layer_ind];
						match &mut layer.y {
							OutputCPU::RNNData(y) => {
								y.update_valid_tpoints(&seq_len_array, max_seq_len);
							} OutputCPU::Tensor(y) => {
								y.update_batch_sz(max_seq_len*batch_sz);
							}
						}
						
					// input is a tensor, output should only be a tensor
					} OutputCPU::Tensor(x) => {
						let batch_sz = x.shape.n;
						
						let layer = &mut self.layers[*layer_ind];
						layer.y.tensor_mut().update_batch_sz(batch_sz);
					}
				}
			}
			
			// run internal forward function
			run_internal_cpu!{&mut self.layer_internals[*layer_ind] => forward(*layer_ind, &mut self.layers) };
			self.layers[*layer_ind].run_fwd = true;
		}
	}
	
	pub fn zero_out_states(&mut self) {
		for layer_internal in self.layer_internals.iter_mut() {
			run_internal_cpu!{layer_internal => zero_out_internal_states() };
		}
	}

	/////////////////////// loading
	pub fn load(model_dir: &str, batch_sz: usize) -> ModelCPU {
		let mut rng = XorState::clock_init();
		
		let key_sets = config_parse(read_file(&format!("{}/{}", model_dir, LAYER_CONFIG_NM)));
		let mut model = ModelCPU::new(batch_sz as c_int);
		
		for layer_keys in key_sets.iter().skip(1) {
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
				//"pooling" => {model.load_max_pooling(layer_keys);}
				//"softmax" => {model.load_softmax(layer_keys);}
				"softmax_log" => {model.load_softmax_log(layer_keys);}
				/*"relu" => {model.load_relu(layer_keys);}
				"add" => {model.load_add(&x_layers, layer_keys);}*/
				"mul" => {model.load_mul(&x_layers, layer_keys);}
				"sum_reduce" => {model.load_sum_reduce(layer_keys);}
				//"pow" => {model.load_pow(&x_layers, layer_keys);}
				"LSTM" => {model.load_lstm(layer_keys);}
				"imgs" => {model.load_imgs(layer_keys);}
				"time_series" => {model.load_time_series(layer_keys);}
				_ => {panic!("unknown layer type: {}", layer_type);}
			}
			
			debug_assert!(x_layers == model.layers.last().unwrap().x_layers);
			
			// load weights
			{
				let layer_ind = model.layers.len() - 1;
				let layer = &model.layers.last().unwrap();
				let file_nm = &format!("{}_{}", layer_ind, layer.nm);
				
				run_internal_cpu!{&mut model.layer_internals[layer_ind] =>
							ld_weights(&format!("{}/weights", model_dir), file_nm) };
			}
		}
		model
	}
	
	pub fn input_layer_inds(&self) -> Vec<usize> {
		let mut inds = Vec::new();
		for (ind, layer_internals) in self.layer_internals.iter().enumerate() {
			match layer_internals {
				InternalTypesCPU::Conv(_) |
				InternalTypesCPU::Softmax(_) |
				InternalTypesCPU::Mul(_) |
				InternalTypesCPU::SumReduce(_) |
				InternalTypesCPU::LSTM(_) => {}
				
				InternalTypesCPU::Img(_) |
				InternalTypesCPU::TimeSeries(_) => {
					inds.push(ind);
				}
			}
		}
		inds
	}
}

