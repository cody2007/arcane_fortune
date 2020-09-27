// trains a transformer one-letter at a time on ~500k quotes
// implementation of a transformer model

// -the first input target (in time) is a null vector;	shape: [dict_sz]
// -the last output target (in time) is dict_sz-1;		shape: [dict_sz]
// 	where dict_sz is the number of the `# of unique encoding symbols in the training
//	dataset` + 1 (i.e., the last output is a symbol that never occurs within the trained quote inputs)

extern crate ironnetpro;
use ironnetpro::*;
use std::os::raw::{c_int};
use std::time::Instant;

const DICT: &str = " ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz.'\"?!0123456789-,:;$()&/*%";

const LOAD_FRM_FILE: bool = false;
//const LOAD_FRM_FILE: bool = true;

const GPU_CARD: c_int = 0;
const BATCH_SZ: usize = 64;//128;
const MAX_SEQ_LEN: usize = 30;//60;//30; //20;//30;//60;//85;//100;

const PERIOD_CHAR: u8 = 53; // period character code for splitting quotes, if they need to be cropped

const WARMUP_STEPS: f32 = 4_000.;
const DATA_TYPE: cudnnDataType_t = cudnnDataType_t::CUDNN_DATA_FLOAT;
const SAVE_FREQ: u64 = 200;

const N_LAYERS: c_int = 4;

const TEMPERATURE: f32 = 1.;//0.1;

const N_HEADS: c_int = 8;
const VEC_IN: c_int = 2*16*N_HEADS;//64*N_HEADS;
const FEED_FORWARD_SZ: c_int = 2*2*256*2;//2*2*256;//2048;//2*2*256;

const LABEL_SMOOTHING: f32 = 0.;//0.1;//1.;//0.05;//.1;

fn ret_eps(step: u64) -> Vec<f32> {
	let step = step as f32;
	
	// lrate = VEC_IN^.5 * min(step^-0.5, step * warmup_steps^-1.5)
	vec![(VEC_IN as f32).powf(-0.5) * ((step.powf(-0.5)).min(step * WARMUP_STEPS.powf(-1.5)))]
}

fn one_hot(t: usize, dict_sz: usize, encoding: &Vec<u8>, label_smoothing: f32) -> Vec<f32> {
	let mut vals = vec!{label_smoothing / (dict_sz as f32); dict_sz};
	
	if encoding.len() > t {
		vals[encoding[t] as usize] += 1. - label_smoothing;
		
	// if string not long enough, fill null values for remainder
	}else{
		vals[dict_sz-1] += 1. - label_smoothing;
	}
	
	vals
}

// encodings: [batch][img][time]
fn letter_batch(batch: usize, dict_sz: usize, encodings: &Vec<Vec<Vec<u8>>>) -> (Vec<f32>, Vec<f32>) {
	let mut inputs = Vec::with_capacity(BATCH_SZ*MAX_SEQ_LEN*dict_sz);
	let mut targets = Vec::with_capacity(BATCH_SZ*MAX_SEQ_LEN*dict_sz);
	
	let batch = batch % encodings.len();
	let encodings_batch = &encodings[batch];
	
	debug_assert!(encodings_batch.len() == BATCH_SZ);
	
	for encoding in encodings_batch.iter() {
		for t in 0..MAX_SEQ_LEN {
			let mut input = if t != 0 {
				one_hot(t-1, dict_sz, encoding, 0.)
			}else{ // first input is null
				vec!{0.; dict_sz}
			};
			
			// position encoding
			/*for (dim, val) in input.iter_mut().enumerate() {
				let rad = t as f32/(10_000_f32.powf((2.*dim as f32) / dict_sz as f32));
				*val = if (dim % 2) == 0 {
					rad.sin()
				}else{
					rad.cos()
				};
			}*/
			
			inputs.append(&mut input);
			targets.append(&mut one_hot(t, dict_sz, encoding, LABEL_SMOOTHING));
		} // t
	} // img
	
	debug_assert!(inputs.len() == (BATCH_SZ*MAX_SEQ_LEN*dict_sz));
	debug_assert!(targets.len() == (BATCH_SZ*MAX_SEQ_LEN*dict_sz));
	(inputs, targets)
}

fn main() {
	let model_dir = format!("/tmp/model_quote_transformer_{}len_{}batchsz_{}layers_{}vec_in_{}ffsz_dpos_w_smoothing_w_scale_weightinit_testbias",
			MAX_SEQ_LEN, BATCH_SZ, N_LAYERS, VEC_IN, FEED_FORWARD_SZ);
	
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
		println!("loading: {}", model_dir);
		Model::load(&model_dir, 0., BATCH_SZ as i32)
	}else{
		println!("creating: {}", model_dir);
		let mut model = Model::new(0., BATCH_SZ as i32);
		
		let imgs_shape = TensorShape {
			n: BATCH_SZ as i32,
			c: MAX_SEQ_LEN as i32,
			h: dict_sz as i32,
			w: 1
		};
		
		model.add_imgs(ImgParams {
					shape: imgs_shape,
					data_type: DATA_TYPE });
		
		model.add_fully_connected_w_bias(FullyConnectedWBiasParams {
				vec_out_sz: VEC_IN,
				weight_initialization: WeightInitialization::XavierUniform(VEC_IN, dict_sz as i32),
				bias_initialization: WeightInitialization::XavierUniform(VEC_IN, 1),
				data_type: DATA_TYPE
		}, &mut rng);
		
		for _ in 0..N_LAYERS {
			model.add_multi_head_attn(MultiHeadAttnParams {
					n_heads: N_HEADS,
					feed_forward_sz: FEED_FORWARD_SZ,
					data_type: DATA_TYPE
				}, &mut rng);
		}
		
		model.add_fully_connected_w_bias(FullyConnectedWBiasParams {
				vec_out_sz: dict_sz as i32,
				weight_initialization: WeightInitialization::XavierUniform(dict_sz as i32, VEC_IN),
				bias_initialization: WeightInitialization::XavierUniform(dict_sz as i32, 1),
				data_type: DATA_TYPE
		}, &mut rng);
		
		// collapse batch and time dimension
		{
			// X[batch_sz, n_time, dict_sz, 1] -> [batch_sz*n_time, dict_sz, 1]
			let x = model.layers[model.layers.len() - 1].y.tensor();
			
			let batch_time = x.shape.n*x.shape.c;
			assert!(x.shape.w == 1);
			
			model.add_transpose_reshape(TransposeReshapeParams {
				fwd_dims: vec![0,1,2,3], // reshape only, do not transpose
				new_shape: TensorShape {
					n: batch_time,
					c: 1,
					h: dict_sz as i32,
					w: 1
				},
				data_type: DATA_TYPE
			});
		}
		
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
		let mut time = Instant::now();
		for i in 0..1_000_000 {
			let (inputs, targets) = letter_batch(i, dict_sz, &encodings);
			//let (inputs, targets) = letter_batch(0, dict_sz, &encodings);
			
			model.layers[input_ind].set_output(&inputs);
			model.layers[labels_ind].set_output(&targets);
			
			model.eps_f32 = ret_eps(model.batch);
			//model.eps_f32 = vec![1e-1 / BATCH_SZ as f32];
			//model.rms_descent(model.layers.len()-1);
			model.adam_descent(model.layers.len()-1);
			
			if (model.batch % SAVE_FREQ) == 0 {
				model.log_err();
				
				/////////// print stats
				let y_output = model.layers[pred_ind].y.ret();
				let err_str = format!("{} err {} corr {}", model.batch,
						model.errs.last().unwrap() / (BATCH_SZ * MAX_SEQ_LEN) as f32, corr(&y_output, &targets));
				
				let pred = model.layers[pred_ind].y.ret(); // from training batch
				
				inputs.sv(&model_dir, "inputs");
				targets.sv(&model_dir, "target");
				model.sv(&model_dir);
				//model.sv_gradients_and_outputs(model_dir);
				
				//////////// get inference predictions (fed outputs)
				let txt_output = {
					model.zero_out_states();
					
					let mut txt_output = Vec::with_capacity(MAX_SEQ_LEN);
					
					for t in 0..MAX_SEQ_LEN {
						// use prior output as input
						let y_input = if t != 0 {
							//////// take max of output across input_sz dim [batch_sz, t, input_sz]
							let mut pred = model.layers[pred_ind].y.ret();
							let mut y_input = model.layers[input_ind].y.ret();
							debug_assert!(pred.len() == (BATCH_SZ*MAX_SEQ_LEN*dict_sz));
							
							// find max across each image, then set max = 1 and all other values to 0
							for img in 0..BATCH_SZ {
								// adjust temperature (and convert to probabilities)
								{
									let offset = img*MAX_SEQ_LEN*dict_sz + (t-1)*dict_sz;
									for p in pred.iter_mut().skip(offset).take(dict_sz) {
										*p = p.exp() / TEMPERATURE;
									}
									
									let sum: f32 = pred.iter().skip(offset).take(dict_sz).sum();
									for p in pred.iter_mut().skip(offset).take(dict_sz) {
										*p /= sum;
									}
								}
								
								// probabilstically sample
								let encoding_ind = {
									let prob_val = rng.gen_f32b();
									let mut max_ind = 0;
									let mut val_sum = 0.;
									
									for ind in 0..dict_sz {
										let ind_use = img*MAX_SEQ_LEN*dict_sz + (t-1)*dict_sz + ind;
										val_sum += pred[ind_use];
										if prob_val < val_sum {
											max_ind = ind;
											break;
										}
									}
									max_ind
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
								y_input[img*MAX_SEQ_LEN*dict_sz + t*dict_sz + encoding_ind] = 1.;
							}
							y_input
							
						// first input in sequence (nulls)
						}else{
							vec!{0.; BATCH_SZ*MAX_SEQ_LEN*dict_sz}
						};
						
						model.layers[input_ind].set_output(&y_input);
						model.reset_fwd_cache_flags();
						model.forward(pred_ind);
					}
					
					model.layers[input_ind].y.ret().sv(&model_dir, "inputs_self_directed");
					model.layers[pred_ind].y.ret().sv(&model_dir, "y_inference_self_directed");
					
					txt_output.into_iter().collect::<String>() // return text output string
				};
				
				pred.sv(&model_dir, "pred");
				
				println!("{} t {} {} {}", err_str, time.elapsed().as_millis() as f32 / 1000., txt_output, model.eps_f32[0]);
				
				time = Instant::now();
				//break;
			}
		}
	}
	
	//write_prof();
}

