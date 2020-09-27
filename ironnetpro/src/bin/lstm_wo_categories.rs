// trains a deep LSTM one-letter at a time on ~500k quotes
// implementation of "Generating Sequences With Recurrent Neural Networks", Alex Graves
//	plotting: http://localhost:8888/notebooks/modeling_notebooks/dataset_gen_letter_embeddings_wo_categories.ipynb
//	dataset: http://localhost:8888/notebooks/modeling_notebooks/training_ironnet_letter_embeddings_wo_categories.ipynb

// -the first input target (in time) is a null vector;	shape: [dict_sz]
// -the last output target (in time) is dict_sz-1;		shape: [dict_sz]
// 	where dict_sz is the number of the `# of unique encoding symbols in the training
//	dataset` + 1 (i.e., the last output is a symbol that never occurs within the trained quote inputs)

extern crate ironnetpro;
use ironnetpro::*;
use std::os::raw::{c_float, c_int};
use std::time::Instant;
use std::cmp::min;

const DICT: &str = " ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz.'\"?!0123456789-,:;$()&/*%";

const LOAD_FRM_FILE: bool = false;
//const LOAD_FRM_FILE: bool = true;

const MODEL_DIR: &str = "/tmp/model_lstm_rms";

const GPU_CARD: c_int = 0;
const BATCH_SZ: usize = 128;
const MAX_SEQ_LEN: usize = 30;//85;//100;
//const EPS: f32 = 5e-4;
const EPS: f32 = 1e-1;

const PERIOD_CHAR: u8 = 53; // period character code for splitting quotes, if they need to be cropped

const DATA_TYPE: cudnnDataType_t = cudnnDataType_t::CUDNN_DATA_FLOAT;
const NORM_SCALE: f32 = 1e-1;
const SAVE_FREQ: u64 = 100;

const N_LAYERS: c_int = 5;
const HIDDEN_SZ: c_int = 128*2*2*2;

const DROPOUT_CHANCE: c_float = 0.;

fn one_hot(t: usize, dict_sz: usize, encoding: &Vec<u8>) -> Vec<f32> {
	let mut vals = vec!{0.; dict_sz};
	
	if encoding.len() > t {
		vals[encoding[t] as usize] = 1.;
	// if string not long enough, fill null values for remainder
	}else{
		vals[dict_sz-1] = 1.;
	}
	
	vals
}

// encodings: [batch][img][time]
fn letter_batch(batch: usize, dict_sz: usize, encodings: &Vec<Vec<Vec<u8>>>
		) -> (Vec<f32>, Vec<f32>, Vec<c_int>) {
	let mut inputs = Vec::with_capacity(MAX_SEQ_LEN*BATCH_SZ*dict_sz);
	let mut targets = Vec::with_capacity(MAX_SEQ_LEN*BATCH_SZ*dict_sz);
	
	let batch = batch % encodings.len();
	let encodings_batch = &encodings[batch];
	
	// length of each sequence
	let mut seq_len_array = Vec::with_capacity(BATCH_SZ);
	debug_assert!(encodings_batch.len() == BATCH_SZ);
	for encoding in encodings_batch.iter() {
		seq_len_array.push(min(MAX_SEQ_LEN as c_int, 1 + encoding.len() as c_int));
	}
	
	for t in 0..MAX_SEQ_LEN {
		for encoding in encodings_batch.iter() {
			if t != 0 {
				inputs.append(&mut one_hot(t-1, dict_sz, encoding));
			}else{ // first input is null
				inputs.append(&mut vec!{0.; dict_sz});
			}
			targets.append(&mut one_hot(t, dict_sz, encoding));
		} // img
	} // t
	
	debug_assert!(inputs.len() == (MAX_SEQ_LEN*BATCH_SZ*dict_sz));
	debug_assert!(targets.len() == (MAX_SEQ_LEN*BATCH_SZ*dict_sz));
	(inputs, targets, seq_len_array)
}

fn main() {
	let dict: Vec<char> = DICT.chars().collect();
	let mut rng = XorState::clock_init();
	
	///////////
	let (encodings, dict_sz) = {
		let encodings_concat = load_numpy::<u8>("/home/tapa/docs/datasets/ironnetpro/quote_letter_encodings.npy");
		
		let dict_sz = (*encodings_concat.iter().max().unwrap() as usize) + 1;
		
		// split up by quote
		let mut encodings_flat = Vec::with_capacity(100_000); // [img][time]
		let mut start_pos = 0;
		for (pos, val) in encodings_concat.iter().enumerate() {
			// termination of quote
			if *val == (dict_sz-1) as u8 {
				let quote = encodings_concat[start_pos..=pos].to_vec();
				
				// quote short enough to keep directly without cropping
				if (pos+1)-start_pos <= MAX_SEQ_LEN {
					encodings_flat.push(quote);
					
				// crop at last period within MAX_SEQ_LEN
				}else{
					for (ind, c) in quote.iter().take(MAX_SEQ_LEN).enumerate().rev() {
						if *c == PERIOD_CHAR {
							encodings_flat.push(quote[..=ind].to_vec());
							break;
						}
					}
				}
				start_pos = pos + 1;
			}
		}
		
		// randomize order
		for i in 0..encodings_flat.len() {
			let j = (rng.gen() as usize) % encodings_flat.len();
			let encoding_back = encodings_flat[i].clone();
			encodings_flat[i] = encodings_flat[j].clone();
			encodings_flat[j] = encoding_back;
		}
		
		// split up by batch, sort within batch for sequences to descend in length
		// output dimension: encodings: [batch][img][time]
		let n_batches = encodings_flat.len() / BATCH_SZ;
		let mut encodings = Vec::with_capacity(n_batches);
		for batch in 0..n_batches {
			let mut encoding_batch = Vec::with_capacity(BATCH_SZ);
			for encoding in encodings_flat.iter().skip(batch*BATCH_SZ).take(BATCH_SZ) {
				encoding_batch.push(encoding.clone());
			}
			
			encoding_batch.sort_unstable_by_key(|v| -(v.len() as isize)); // sort from greatest to least length
			encodings.push(encoding_batch);
		}
		
		println!("n_batches: {} dictionary size: {}", encodings.len(), dict_sz);
		(encodings, dict_sz)
	};
	
	/////////////
	set_device(GPU_CARD);
	
	let mut model = if LOAD_FRM_FILE {
		println!("loading: {}", MODEL_DIR);
		Model::load(MODEL_DIR, 0., BATCH_SZ as i32)
	}else{
		let mut model = Model::new(0., BATCH_SZ as i32);
		
		model.add_time_series(TimeSeriesParams {
					max_seq_len: MAX_SEQ_LEN as c_int,
					batch_sz: BATCH_SZ as c_int,
					vec_sz: dict_sz as c_int,
					data_type: DATA_TYPE
				});
		
		model.add_lstm(LSTMParams {
					hidden_sz: HIDDEN_SZ,
					n_layers: N_LAYERS,
					dropout_chance: DROPOUT_CHANCE,
					norm_scale: NORM_SCALE,
					data_type: DATA_TYPE
				}, &mut rng);
		
		model.add_conv(ConvParams {
				n_filters: dict_sz as c_int,
				filter_sz_h: HIDDEN_SZ,
				filter_sz_w: 1, 
				pad_h: 0, 
				pad_w: 0,
				stride: 1,
				norm_scale: NORM_SCALE,
				data_type: DATA_TYPE
			}, &mut rng);
		
		model.add_softmax_cross_entropy_loss(DATA_TYPE);
		
		model
	};
	
	let (input_ind, labels_ind, pred_ind) = {
		let input_inds = model.input_layer_inds();
		let pred_inds = model.find_layer_inds("softmax_log");
		
		assert!(input_inds.len() == 2);
		assert!(pred_inds.len() == 1);
		
		(input_inds[0], input_inds[1], pred_inds[0])
	};
	
	//////// training
	{
		let t_stride = BATCH_SZ*dict_sz;
		let mut time = Instant::now();
		for i in 0..1_000_000 {
			let (inputs, targets, seq_len_array) = letter_batch(i, dict_sz, &encodings);
			
			model.layers[input_ind].set_output_seq(&inputs, &seq_len_array);
			model.layers[labels_ind].set_output_seq(&targets, &seq_len_array);
			
			model.eps_f32 = vec![EPS / BATCH_SZ as f32];//ret_eps(model.batch);
			model.rms_descent(model.layers.len()-1);
			//model.adam_descent(model.layers.len()-1);
			
			if (model.batch % SAVE_FREQ) == 0 {
				model.log_err();
				
				/////////// print stats
				let y_output = model.layers[pred_ind].y.ret();
				let err_str = format!("{} err {} corr {}", model.batch,
						model.errs.last().unwrap() / (BATCH_SZ * MAX_SEQ_LEN) as f32, corr(&y_output, &targets));
				
				////////// get inference predictions (fed correct inputs)
				{
					model.zero_out_states();
					
					let mut y_inference = Vec::with_capacity(MAX_SEQ_LEN);
					for t in 0..MAX_SEQ_LEN {
						model.layers[input_ind].set_output_seq(&inputs[t*t_stride..(t+1)*t_stride].to_vec(), &vec!{1; BATCH_SZ});
						model.reset_fwd_cache_flags();
						model.forward_inference(pred_ind);
						
						y_inference.append(&mut model.layers[pred_ind].y.ret());
					}
					
					y_inference.sv(MODEL_DIR, "y_inference");
				}
				
				//////////// get inference predictions (fed outputs)
				let txt_output = {
					model.zero_out_states();
					
					let mut y_inference = Vec::with_capacity(MAX_SEQ_LEN*BATCH_SZ*dict_sz); // outputs
					let mut inputs_self_directed = Vec::with_capacity(MAX_SEQ_LEN*BATCH_SZ*dict_sz); // inputs
					let mut txt_output = Vec::with_capacity(MAX_SEQ_LEN);
					
					for t in 0..MAX_SEQ_LEN {
						// use prior output as input
						let mut y_input = if t != 0 {
							//////// take max of output across input_sz dim [batch_sz, input_sz]
							let mut y_input = model.layers[pred_ind].y.ret();
							debug_assert!(y_input.len() == (BATCH_SZ*dict_sz));
							
							// find max across each image, then set max = 1 and all other values to 0
							for img in 0..BATCH_SZ {
								// probabilstically sample
								let (max_ind, encoding_ind) = {
									let prob_val = rng.gen_f32b();
									let mut max_ind = 0;
									let mut val_sum = 0.;
									
									for ind in 0..dict_sz {
										let ind_use = img*dict_sz + ind;
										val_sum += y_input[ind_use].exp();
										if prob_val < val_sum {
											max_ind = ind_use;
											break;
										}
									}
									(max_ind, max_ind - img*dict_sz)
								};
								// (max_ind indexes pred, encoding_ind indexes dict)
								
								// text output string
								if img == 0 {
									// sequence termination character
									txt_output.push(if encoding_ind == (dict_sz-1) {
										'\\'
									}else{
										dict[encoding_ind]
									});
								}
								
								// set vals
								for ind in 0..dict_sz {
									let ind_use = img*dict_sz + ind;
									y_input[ind_use] = if ind_use != max_ind {
										0.
									}else{
										1.
									};
								}
							}
							y_input
							
						// first input in sequence (nulls)
						}else{
							vec!{0.; t_stride}
						};
						
						model.layers[input_ind].set_output_seq(&y_input, &vec!{1; BATCH_SZ});
						model.reset_fwd_cache_flags();
						model.forward_inference(pred_ind);
						
						inputs_self_directed.append(&mut y_input);
						y_inference.append(&mut model.layers[pred_ind].y.ret());
					}
					y_inference.sv(MODEL_DIR, "y_inference_self_directed");
					inputs_self_directed.sv(MODEL_DIR, "inputs_self_directed");
					
					txt_output.into_iter().collect::<String>() // return text output string
				};
				
				inputs.sv(MODEL_DIR, "inputs");
				targets.sv(MODEL_DIR, "target");
				model.sv(MODEL_DIR);
				
				println!("{} t {} {}", err_str, time.elapsed().as_millis() as f32 / 1000., txt_output);
				
				time = Instant::now();
			}
		}
	}
}

