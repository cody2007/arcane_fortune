// trains a deep LSTM one-letter at a time on ~500k quotes
// WITH additional inputs denoting the category of the sequence (given at each timestep)
// implementation of "Generating Sequences With Recurrent Neural Networks", Alex Graves
//	plotting: http://localhost:8888/notebooks/modeling_notebooks/dataset_gen_letter_embeddings_w_categories.ipynb
//	dataset: http://localhost:8888/notebooks/modeling_notebooks/training_ironnet_letter_embeddings_w_categories.ipynb
//
// -the first input target (in time) is a null vector;	shape: [N_CATEGORIES + dict_sz]
// -the last output target (in time) is dict_sz-1;	      shape: [dict_sz]
// 	where dict_sz is the number of the `# of unique encoding symbols in the training
//	dataset` + 1 (i.e., the last output is a symbol that never occurs within the trained quote inputs)

extern crate ironnetpro;
use ironnetpro::*;
use std::os::raw::{c_int, c_float};
use std::time::Instant;
use std::cmp::min;

const DICT: &str = " ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz.'\"?!0123456789-,:;$()&/*%";
const N_CATEGORIES: usize = 3;

const LOAD_FRM_FILE: bool = false;
//const LOAD_FRM_FILE: bool = true;

const MAX_SEQ_LEN: usize = 140; // 60
const MODEL_DIR: &str = "/tmp/model140";

const GPU_CARD: c_int = 0;
const BATCH_SZ: usize = 128;

const PERIOD_CHAR: u8 = 53; // period character code for splitting quotes, if they need to be cropped

const EPS: f32 = 1e-3;
const DATA_TYPE: cudnnDataType_t = cudnnDataType_t::CUDNN_DATA_FLOAT;
const NORM_SCALE: f32 = 1e-1;
const SAVE_FREQ: u64 = 100;

const N_LAYERS: c_int = 7;
const HIDDEN_SZ: c_int = 128*2*2*2;

const DROPOUT_CHANCE: c_float = 0.;

// encoding: [MAX_SEQ_LEN] (or less)
// output: [dict_sz]
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

////////////////////
// inputs:
//	encodings: [batch][img][time]
//	categories: [batch][img]
//
// output: 
//	(inputs, targets, seq_len_array)
//	inputs [MAX_SEQ_LEN, BATCH_SZ, (N_CATEGORIES + dict_sz)]
//	targets: [MAX_SEQ_LEN, BATCH_SZ, dict_sz] 
//	seq_len_array: [BATCH_SZ]
fn letter_batch(batch: usize, dict_sz: usize, encodings: &Vec<Vec<Vec<u8>>>,
			categories: &Vec<Vec<u8>>) -> (Vec<f32>, Vec<f32>, Vec<c_int>) {
	let mut inputs = Vec::with_capacity(MAX_SEQ_LEN*BATCH_SZ*(N_CATEGORIES + dict_sz));
	let mut targets = Vec::with_capacity(MAX_SEQ_LEN*BATCH_SZ*dict_sz);
	
	let batch = batch % encodings.len();
	let encodings_batch = &encodings[batch];
	let categories_batch = &categories[batch];
	
	// length of each sequence
	let seq_len_array = {
		let mut seq_len_array = Vec::with_capacity(BATCH_SZ);
		debug_assert!(encodings_batch.len() == BATCH_SZ);
		for encoding in encodings_batch.iter() {
			seq_len_array.push(min(MAX_SEQ_LEN as c_int, 1 + encoding.len() as c_int));
		}
		seq_len_array
	};
	
	for t in 0..MAX_SEQ_LEN {
		// loop across batches
		for (encoding, category) in encodings_batch.iter().zip(categories_batch.iter()) {
			// push category
			{
				let mut category_encoding = vec!{0.; N_CATEGORIES};
				category_encoding[*category as usize] = 1.;
				
				inputs.append(&mut category_encoding);
			}
			
			// push quote encoding
			if t != 0 {
				inputs.append(&mut one_hot(t-1, dict_sz, encoding));
			}else{ // first input is null
				inputs.append(&mut vec!{0.; dict_sz});
			}
			targets.append(&mut one_hot(t, dict_sz, encoding));
		} // img
	} // t
	
	debug_assert!(inputs.len() == MAX_SEQ_LEN*BATCH_SZ*(N_CATEGORIES + dict_sz), "{}", inputs.len());
	debug_assert!(targets.len() == MAX_SEQ_LEN*BATCH_SZ*dict_sz, "{}", targets.len());
	(inputs, targets, seq_len_array)
}

fn main() {
	let dict: Vec<char> = DICT.chars().collect();
	let mut rng = XorState::clock_init();
	
	///////////
	let (encodings, categories, dict_sz) = {
		let encodings_concat = load_numpy::<u8>("/home/tapa/docs/datasets/ironnetpro/quote_letter_encodings.npy");
		let categories = load_numpy::<u8>("/home/tapa/docs/datasets/ironnetpro/quote_letter_encodings_categories.npy");
		
		let dict_sz = (*encodings_concat.iter().max().unwrap() as usize) + 1;
		assert!((*categories.iter().max().unwrap() as usize) < N_CATEGORIES);
		
		// split up by quote
		let mut encodings_flat = Vec::with_capacity(100_000); // [img][time]
		let mut categories_flat = Vec::with_capacity(100_000);
		
		let mut start_pos = 0;
		for (pos, val) in encodings_concat.iter().enumerate() {
			// termination of quote
			if *val == (dict_sz-1) as u8 {
				let quote = encodings_concat[start_pos..=pos].to_vec();
				
				// quote short enough to keep directly without cropping
				if (pos+1)-start_pos <= MAX_SEQ_LEN {
					encodings_flat.push(quote);
					categories_flat.push(categories[encodings_flat.len()-1]);
					
				// crop at last period within MAX_SEQ_LEN
				}else{
					for (ind, c) in quote.iter().take(MAX_SEQ_LEN).enumerate().rev() {
						if *c == PERIOD_CHAR {
							encodings_flat.push(quote[..=ind].to_vec());
							categories_flat.push(categories[encodings_flat.len()-1]);
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
			let categories_back = categories_flat[i].clone();
			
			encodings_flat[i] = encodings_flat[j].clone();
			encodings_flat[j] = encoding_back;
			
			categories_flat[i] = categories_flat[j].clone();
			categories_flat[j] = categories_back;
		}
		
		// split up by batch, sort within batch for sequences to descend in length
		// output dimension: encodings: [batch][img][time]
		let n_batches = encodings_flat.len() / BATCH_SZ;
		let mut encodings = Vec::with_capacity(n_batches);
		let mut categories = Vec::with_capacity(n_batches);
		for (encodings_flatv, categories_flatv) in encodings_flat.chunks(BATCH_SZ)
								.zip(categories_flat.chunks(BATCH_SZ)) {
			
			// don't include the end when a full batch can't be created
			if encodings_flatv.len() < BATCH_SZ {
				break;
			}
			
			struct Exemplar {
				encoding: Vec<u8>,
				category: u8
			}
			
			let sorted_exemplars = {
				let mut exemplars = Vec::with_capacity(BATCH_SZ);
				
				// put encoding and category in one structure for sorting
				for (encoding, category) in encodings_flatv.iter().zip(categories_flatv.iter()) {
					exemplars.push(Exemplar {encoding: encoding.to_vec(), category: *category});
				}
				
				exemplars.sort_unstable_by_key(|v| -(v.encoding.len() as isize)); // sort from greatest to least length
				exemplars
			};
			
			let mut encoding_batch = Vec::with_capacity(BATCH_SZ);
			let mut category_batch = Vec::with_capacity(BATCH_SZ);
			
			for exemplar in sorted_exemplars {
				encoding_batch.push(exemplar.encoding);
				category_batch.push(exemplar.category);
			}
			
			encodings.push(encoding_batch);
			categories.push(category_batch);
		}
		
		println!("n_batches: {} dictionary size: {}", encodings.len(), dict_sz);
		(encodings, categories, dict_sz)
	};
	
	/////////////
	set_device(GPU_CARD);
	
	let mut model = if LOAD_FRM_FILE {
		println!("loading: {}", MODEL_DIR);
		Model::load(MODEL_DIR, EPS / BATCH_SZ as f32, BATCH_SZ as i32)
	}else{
		let mut model = Model::new(EPS / BATCH_SZ as f32, BATCH_SZ as i32);
		
		model.add_time_series(TimeSeriesParams {
					max_seq_len: MAX_SEQ_LEN as c_int,
					batch_sz: BATCH_SZ as c_int,
					vec_sz: (N_CATEGORIES + dict_sz) as c_int,
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
	
	assert!((dict.len() + 1) == dict_sz); // last encoding slot indicates sequence termination
	
	//////// training
	{
		let mut time = Instant::now();
		let input_t_stride = BATCH_SZ*(N_CATEGORIES + dict_sz);
		for i in 0..1_000_000 {
			let (inputs, targets, seq_len_array) = letter_batch(i, dict_sz, &encodings, &categories);
			
			model.layers[input_ind].set_output_seq(&inputs, &seq_len_array);	
			model.layers[labels_ind].set_output_seq(&targets, &seq_len_array);
			
			model.rms_descent(model.layers.len()-1);
			
			if (model.batch % SAVE_FREQ) == 0 {
				model.log_err();
				
				/*if (i / SAVE_FREQ) == 50 || (i / SAVE_FREQ) == 100 {
					println!("reducing learning rate");
					model.eps_f32[0] /= 10.;
				}*/
				
				/////////// print stats
				let y_output = model.layers[pred_ind].y.ret();
				let string_print = format!("{} err {} corr {}", model.batch,
								model.errs.last().unwrap() / (BATCH_SZ * MAX_SEQ_LEN) as f32,
								corr(&y_output, &targets));
				
				////////// get inference predictions (fed correct inputs)
				{
					model.zero_out_states();
					
					let mut y_inference = Vec::with_capacity(MAX_SEQ_LEN);
					for t in 0..MAX_SEQ_LEN {
						model.layers[input_ind].set_output_seq(&inputs[t*input_t_stride..(t+1)*input_t_stride].to_vec(), &vec!{1; BATCH_SZ});
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
					let mut txt_output = Vec::with_capacity(N_CATEGORIES); // example output for each category in string form [N_CATEGORIES][MAX_SEQ_LEN] 
					for _ in 0..N_CATEGORIES { // init txt_output
						txt_output.push(Vec::with_capacity(MAX_SEQ_LEN));
					}
					
					for t in 0..MAX_SEQ_LEN {
						// use prior output as input
						let mut y_input = if t != 0 {
							//////// take max of output across input_sz dim [batch_sz, input_sz]
							let y_output = model.layers[pred_ind].y.ret();
							debug_assert!(y_output.len() == BATCH_SZ*dict_sz, "y_input len {}", y_output.len());
							let mut y_input = vec!{0.; BATCH_SZ*(N_CATEGORIES + dict_sz)};
							
							// find max across each image, then set max = 1 and all other values to 0
							for img in 0..BATCH_SZ {
								// probabilstically sample
								let encoding_ind = {
									let prob_val = rng.gen_f32b();
									let mut encoding_ind = 0;
									let mut val_sum = 0.;
									
									for (i, yv) in y_output.iter()
												.skip(img*dict_sz)
												.take(dict_sz).enumerate() {
										val_sum += yv.exp();
										if prob_val < val_sum {
											encoding_ind = i;
											break;
										}
									}
									encoding_ind
								};
								
								// set category val
								let category_use = img % N_CATEGORIES;
								for (cat, y_inputv) in y_input.iter_mut()
												.skip(img*(N_CATEGORIES + dict_sz))
												.take(N_CATEGORIES).enumerate() {
									*y_inputv = if cat != category_use {0.} else {1.};
								}
								
								// set quote input vals
								for i in 0..dict_sz {
									let ind_use = img*(N_CATEGORIES + dict_sz) + N_CATEGORIES + i;
									
									y_input[ind_use] = if i != encoding_ind {
										0.
									}else{
										1.
									};
								}
								
								// save character for printing
								if img == category_use {
									txt_output[category_use].push(
										if encoding_ind == (dict_sz-1) { // sequence termination character
											'\\'
										}else{
											dict[encoding_ind]
										}
									);
								}
							}
							y_input
							
						// first input in sequence (nulls)
						}else{
							let mut y_input = vec!{0.; BATCH_SZ*(N_CATEGORIES + dict_sz)};
							
							// set category vals
							for img in 0..BATCH_SZ {
								let category_use = img % N_CATEGORIES;
								y_input[img*(N_CATEGORIES + dict_sz) + category_use] = 1.;
							}
							y_input
						};
						
						model.layers[input_ind].set_output_seq(&y_input, &vec!{1; BATCH_SZ});
						model.reset_fwd_cache_flags();
						model.forward_inference(pred_ind);
						
						inputs_self_directed.append(&mut y_input);
						y_inference.append(&mut model.layers[pred_ind].y.ret());
					}
					y_inference.sv(MODEL_DIR, "y_inference_self_directed");
					inputs_self_directed.sv(MODEL_DIR, "inputs_self_directed");
					
					txt_output
				};
				
				inputs.sv(MODEL_DIR, "inputs");
				targets.sv(MODEL_DIR, "target");
				
				model.sv(MODEL_DIR);
				
				// print errs w/ inference text
				{
					let string_print = format!("{} t {} sample self-directed seq:", 
							string_print, time.elapsed().as_millis() as f32 / 1000.);
					
					let mut spacer = String::with_capacity(string_print.len());
					for _ in 0..string_print.len() {spacer.push(' ');}
					
					// print training errs & first category txt prediction
					println!("{} \"{}\"", string_print, txt_output[0].iter().collect::<String>());
					
					// print remaining category texts
					for txt in txt_output.iter().skip(1) {
						println!("{} \"{}\"", spacer, txt.iter().collect::<String>());
					}
				}
				
				time = Instant::now();
			}
		}
	}
}

