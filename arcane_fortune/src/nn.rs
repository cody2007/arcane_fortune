extern crate ironnetpro;
use ironnetpro::*;
use std::time::Instant;
use crate::renderer::endwin;

const DICT: &str = " ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz.'\"?!0123456789-,:;$()&/*%";
const MODEL_DIR: &str = "nn/model_lstm_quotes_w_categories_5layers_512hsz_60seqlen/";

// runs a deep LSTM one-letter at a time inference mode on CPU
// WITH additional inputs denoting the category of the sequence (given at each timestep)
// implementation of "Generating Sequences With Recurrent Neural Networks", Alex Graves

// -the first input target (in time) is a null vector;	shape: [N_CATEGORIES + dict_sz]
// -the last output target (in time) is dict_sz-1;	      shape: [dict_sz]
// 	where dict_sz is the number of the `# of unique encoding symbols in the training
//	dataset` + 1 (i.e., the last output is a symbol that never occurs within the trained quote inputs)

#[derive(Copy, Clone)]
pub enum TxtCategory {
	Neutral = 0,
	Positive = 1,
	Negative = 2
}

use crate::ai::AIPersonality;
impl From<&AIPersonality> for TxtCategory {
	fn from(personality: &AIPersonality) -> Self {
		const SPACE: f32 = 2./3.;
		if personality.friendliness < (SPACE - 1.) {
			TxtCategory::Negative
		}else if personality.friendliness < (2.*SPACE - 1.) {
			TxtCategory::Neutral
		}else{
			TxtCategory::Positive
		}
	}
}

const N_CATEGORIES: usize = 3;
const BATCH_SZ: usize = 1;

//const MAX_SEQ_LEN: usize = 10;//85;//100;
const MAX_SEQ_LEN: usize = 60;

// returns characters or entire strings from the category
pub struct TxtGenerator {
	model: ModelCPU,
	input_ind: usize, // input layer of model
	pred_ind: usize, // output layer of model
	category: TxtCategory,
	dict_sz: usize,
	t: usize,
	dict: Vec<char>,
	rng: XorState
}

impl TxtGenerator {
	pub fn new(seed: u32) -> Self {
		let model = ModelCPU::load(MODEL_DIR, BATCH_SZ);
		
		let (input_ind, pred_ind) = {
			let input_inds = model.input_layer_inds();
			let pred_inds = model.find_layer_inds("softmax_log");
			
			assertq!(input_inds.len() == 2);
			assertq!(pred_inds.len() == 1);
			
			(input_inds[0], pred_inds[0])
		};
		
		TxtGenerator {
			model,
			input_ind,
			pred_ind,
			category: TxtCategory::Neutral,
			dict_sz: DICT.len() + 1, // a network output encoding of `dict_sz-1` indicates the network finished the quote
			t: 0,
			dict: DICT.chars().collect(),
			rng: XorState::init(seed as u64)
		}
	}
	
	// set TxtGenerator state to generate new sequence in category type
	pub fn new_seq(&mut self, category: TxtCategory) {
		self.t = 0;
		self.category = category;
		self.model.zero_out_states();
	}
	
	// generate single character
	pub fn gen_char(&mut self) -> Option<char> {
		if self.t >= MAX_SEQ_LEN {
			return None;
		}
		
		let mut y_input = vec!{0.; N_CATEGORIES + self.dict_sz};
		
		// set category vals
		y_input[self.category as usize] = 1.;
		
		// run forward with null inputs to get initial letter prediction
		if self.t == 0 {
			self.model.layers[self.input_ind].set_output_seq(y_input.clone(), &vec!{1; BATCH_SZ});
			self.model.reset_fwd_cache_flags();
			self.model.forward(self.pred_ind);
		}
		
		self.t += 1;
		
		// prediction output from previous timestep
		let pred = {
			let pred_output = &self.model.layers[self.pred_ind].y;
			let n_elements = pred_output.ravel_time_shape().n_elements();
			pred_output.mem()[..n_elements].to_vec()
		};
		debug_assertq!(pred.len() == self.dict_sz);
		
		// use prior output as input, save output for printing
		// find max across each image, save output for printing, then set max = 1 and all other values to 0
		
		// probabilstically sample
		let encoding_ind = {
			let prob_val = self.rng.gen_f32b();
			let mut encoding_ind = 0;
			let mut val_sum = 0.;
			
			for (i, predv) in pred.iter().enumerate() {
				val_sum += predv.exp();
				if prob_val < val_sum {
					encoding_ind = i;
					break;
				}
			}
			encoding_ind
		};
		
		// set quote input vals
		y_input[N_CATEGORIES + encoding_ind] = 1.;
		
		// return character for printing
		return if encoding_ind == (self.dict_sz-1) { // sequence termination character
			None
		}else{
			// run forward for next value
			if self.t < MAX_SEQ_LEN {
				self.model.layers[self.input_ind].set_output_seq(y_input, &vec!{1; BATCH_SZ});
				self.model.reset_fwd_cache_flags();
				self.model.forward(self.pred_ind);
			}
			
			Some(self.dict[encoding_ind])
		};
	}
	
	// generate and return entire string
	pub fn gen_str(&mut self, category: TxtCategory) -> String {
		self.new_seq(category);
		
		let mut txt = Vec::with_capacity(MAX_SEQ_LEN);
		for _ in 0..MAX_SEQ_LEN {
			if let Some(c) = self.gen_char() {
				txt.push(c);
			}else{
				break;
			}
		}
		
		txt.iter().collect::<String>()
	}
}

// intended to be called in printing code where it is desirable to have
// letters printed as they are generated. it will attempt to generate as many
// as it can within MAX_TIME_USAGE. and on the next call will generate more
// to fill out the remainder of the sequence
const MAX_TIME_USAGE: f32 = 200.; // milliseconds
pub struct TxtPrinter {
	txt_gen: TxtGenerator,
	seq: Vec<char>,
	finished: bool
}

impl TxtPrinter {
	pub fn new(category: TxtCategory, seed: u32) -> Self {
		let mut txt_gen = TxtGenerator::new(seed);
		txt_gen.category = category;
		
		TxtPrinter {
			txt_gen,
			seq: Vec::new(),
			finished: false
		}
	}
	
	pub fn gen(&mut self) -> String {
		macro_rules! ret_seq{() => {return self.seq.iter().collect::<String>();}}
		if self.finished {ret_seq!();}
		
		let t_start = Instant::now();
		loop {
			// add character to buffer
			if let Some(ch) = self.txt_gen.gen_char() {
				self.seq.push(ch);
				
			// finished sequence generation
			}else{
				self.finished = true;
				break;
			}
			
			// out of time
			if (t_start.elapsed().as_millis() as f32) > MAX_TIME_USAGE {
				break;
			}
		}
		ret_seq!();
	}
	
	// set TxtPrinter state to generate new sequence in the same category type
	pub fn new_seq(&mut self) {
		self.seq.clear();
		self.finished = false;
		self.txt_gen.new_seq(self.txt_gen.category);
	}
}

