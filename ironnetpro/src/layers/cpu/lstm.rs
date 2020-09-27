use std::os::raw::c_int;
use super::*;
use crate::saving::{KeyPair, find_req_key_parse};
use crate::layers::LSTMParams;

#[allow(non_snake_case)]
struct LSTMWeights {
	Wi: Filter3CPU, bWi: Filter3CPU,
	Wf: Filter3CPU, bWf: Filter3CPU,
	Wc: Filter3CPU, bWc: Filter3CPU,
	Wo: Filter3CPU, bWo: Filter3CPU,
	
	Ri: Filter3CPU, bRi: Filter3CPU,
	Rf: Filter3CPU, bRf: Filter3CPU,
	Rc: Filter3CPU, bRc: Filter3CPU,
	Ro: Filter3CPU, bRo: Filter3CPU
}

struct State { // [batch_sz, params.hidden_sz]
	h: TensorCPU,
	c: TensorCPU
}

struct PseudoLayer {
	weights: LSTMWeights,
	state: State
}

pub struct LSTMInternalsCPU {
	params: LSTMParams,
	batch_sz: c_int,
	pseudo_layers: Vec<PseudoLayer>
}

impl RunCPU for LSTMInternalsCPU {
	fn forward(&mut self, layer_ind: usize, layers: &mut Vec<LayerCPU>) {
		#[cfg(feature="profile")]
		let _g = Guard::new("lstm cpu forward");
		
		let layer = &layers[layer_ind];
		debug_assert!(layer.x_layers.len() == 1);
		
		let mut x = layers[layer.x_layers[0]].y.rnn_data().clone(); // output of input layer is the input for this layer
		
		debug_assert!(*x.seq_len_array.iter().max().unwrap() == 1,
				"lstm cpu forward only implemented when run one timestep at a time");
		
		for pseudo_layer_ind in 0..self.params.n_layers as usize {
			// see cudnnAPI: DA-09702-001_v7.6.5 | 50
			//
			// i = σ(Wi*x + Ri*ht-1 + bWi + bRi)
			// f = σ(Wf*x + Rf*ht-1 + bWf + bRf)
			// o = σ(Wo*x + Ro*ht-1 + bWo + bRo)
			// c't = tanh(Wc*xt + Rc*ht-1 + bWc + bRc)
			// ct = f ◦ ct-1 + i ◦ c't
			// ht = o ◦ tanh(ct)
			//
			// where "◦" denotes point-wise multiplication
			
			let pl = &self.pseudo_layers[pseudo_layer_ind];
			
			let w = &pl.weights;
			let h = &pl.state.h;
			let c = &pl.state.c;
			
			// output = W*x + R*ht-1 + bW + bR
			#[allow(non_snake_case)]
			let gate_internal = |W: &Filter3CPU, R: &Filter3CPU, bW: &Filter3CPU, bR: &Filter3CPU| -> RNNDataCPU {
				W.mat_mul_rnn(&x) + R.mat_mul_rnn(h) + bW + bR
			};
			
			#[cfg(feature="profile")]
			let gate = Guard::new("gate");
			
			let i =  gate_internal(&w.Wi, &w.Ri, &w.bWi, &w.bRi).sigmoid();
			let f =  gate_internal(&w.Wf, &w.Rf, &w.bWf, &w.bRf).sigmoid();
			let o =  gate_internal(&w.Wo, &w.Ro, &w.bWo, &w.bRo).sigmoid();
			let c_ = gate_internal(&w.Wc, &w.Rc, &w.bWc, &w.bRc).tanh();
			
			#[cfg(feature="profile")]
			drop(gate);
			
			///////// update internal states
			#[cfg(feature="profile")]
			let update = Guard::new("update");
			
			// ct = f ◦ ct-1 + i ◦ c't
			self.pseudo_layers[pseudo_layer_ind].state.c = TensorCPU::from(f*c + i*c_);
			
			// ht = o ◦ tanh(ct)
			let c = &self.pseudo_layers[pseudo_layer_ind].state.c;
			self.pseudo_layers[pseudo_layer_ind].state.h = TensorCPU::from(o * c.tanh());
			
			x = RNNDataCPU::from(self.pseudo_layers[pseudo_layer_ind].state.h.clone());
		}
		
		layers[layer_ind].y.rnn_data_mut().mem = x.mem;
	}
	
	fn zero_out_internal_states(&mut self) {
		for pseudo_layer in self.pseudo_layers.iter_mut() {
			pseudo_layer.state.h.zero_out();
			pseudo_layer.state.c.zero_out();
		}
	}
	
	fn ld_weights(&mut self, save_dir: &str, file_nm: &str) {
		let n_pseudo_layers = self.params.n_layers as usize;
		self.pseudo_layers = Vec::with_capacity(n_pseudo_layers);
		
		let state_shape = Tensor3Shape {
						dim1: self.batch_sz,
						dim2: self.params.hidden_sz,
						dim3: 1
		};
		
		for pseudo_layer in 0..n_pseudo_layers { // hidden layer number
			let file_nm_prefix = format!("{}/{}_{}", save_dir, file_nm, pseudo_layer);
			let ld_vals = |weights_nm| {
				Filter3CPU::load_numpy(&format!("{}_{}.npy", file_nm_prefix, weights_nm))
			};
			
			self.pseudo_layers.push( PseudoLayer {
				weights: LSTMWeights {
					Wi: ld_vals("Wi"), bWi: ld_vals("bWi"),
					Wf: ld_vals("Wf"), bWf: ld_vals("bWf"),
					Wc: ld_vals("Wc"), bWc: ld_vals("bWc"),
					Wo: ld_vals("Wo"), bWo: ld_vals("bWo"),
					
					Ri: ld_vals("Ri"), bRi: ld_vals("bRi"),
					Rf: ld_vals("Rf"), bRf: ld_vals("bRf"),
					Rc: ld_vals("Rc"), bRc: ld_vals("bRc"),
					Ro: ld_vals("Ro"), bRo: ld_vals("bRo")
				},
				state: State {
					h: TensorCPU::zeros3(&state_shape),
					c: TensorCPU::zeros3(&state_shape)
				}
			});
		}
	}
}

impl ModelCPU {
	pub fn add_lstm(&mut self, params: LSTMParams) {
		debug_assert!(self.layers.len() > 0);
		
		let layer_prev_ind = self.layers.len() - 1;
		let layer_prev = &self.layers[layer_prev_ind];
		
		let x = layer_prev.y.rnn_data(); // (input to this layer is the output of the previous layer)
		let y = RNNDataCPU::new(x.max_seq_len, x.batch_sz, params.hidden_sz, &x.seq_len_array);
		
		let batch_sz = x.batch_sz;
		
		self.new_layer_time_series(
			vec!{layer_prev_ind; 1},
			InternalTypesCPU::LSTM(LSTMInternalsCPU {
					params,
					batch_sz,
					pseudo_layers: Vec::new(),
			}),
			y,
			String::from("LSTM"),
		);
	}
	
	pub fn load_lstm(&mut self, layer_keys: &Vec<KeyPair>) {
		let data_type = find_req_key_parse("data_type", &layer_keys);
		debug_assert!(data_type == cudnnDataType_t::CUDNN_DATA_FLOAT);
		
		self.add_lstm(LSTMParams {
			hidden_sz: find_req_key_parse("hidden_sz", layer_keys),
			n_layers: find_req_key_parse("n_layers", layer_keys),
			dropout_chance: find_req_key_parse("dropout_chance", layer_keys),
			norm_scale: find_req_key_parse("norm_scale", layer_keys),
			data_type
		});
	}
}

