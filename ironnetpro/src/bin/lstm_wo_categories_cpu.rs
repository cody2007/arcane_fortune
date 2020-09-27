// runs the deep LSTM on the CPU from a pre-trained model
// implementation of "Generating Sequences With Recurrent Neural Networks", Alex Graves
//	plotting: http://localhost:8888/notebooks/modeling_notebooks/dataset_gen_letter_embeddings_wo_categories.ipynb
//	dataset: http://localhost:8888/notebooks/modeling_notebooks/training_ironnet_letter_embeddings_wo_categories.ipynb

// -the first input target (in time) is a null vector len: [dict_sz]
// -the last output target (in time) is dict_sz-1; shape: [dict_sz]
// 	where dict_sz is the number of the `# of unique encoding symbols in the training
//	dataset` + 1 (i.e., the last output is a symbol that never occurs within the trained quote inputs)

extern crate ironnetpro;
use ironnetpro::*;
use std::time::Instant;

const DICT: &str = " ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz.'\"?!0123456789-,:;$()&/*%";
const MODEL_DIR: &str = "/tmp/model";

const BATCH_SZ: usize = 1;
const MAX_SEQ_LEN: usize = 60;

fn main() {
	let dict: Vec<char> = DICT.chars().collect();
	let mut rng = XorState::clock_init();
	
	/////////////
	let mut model_cpu = ModelCPU::load(MODEL_DIR, BATCH_SZ);
	
	let (input_ind, pred_ind) = {
		let input_inds = model_cpu.input_layer_inds();
		let pred_inds = model_cpu.find_layer_inds("softmax_log");
		
		assert!(input_inds.len() == 2);
		assert!(pred_inds.len() == 1);
		
		(input_inds[0], pred_inds[0])
	};
	
	let dict_sz = DICT.len() + 1; // a network output encoding of `dict_sz-1` indicates the network finished the quote	
	let t_stride = BATCH_SZ*dict_sz;
	
	model_cpu.zero_out_states();
	
	let time = Instant::now();	
	let mut txt_output = Vec::with_capacity(MAX_SEQ_LEN);
	
	for t in 0..MAX_SEQ_LEN {
		let mut pred = {
			let pred_output = &model_cpu.layers[pred_ind].y;
			let n_elements = pred_output.ravel_time_shape().n_elements();
			pred_output.mem()[..n_elements].to_vec()
		};
		
		// use prior output as input, save output for printing
		let y_input = if t != 0 {
			//////// take max of output across input_sz dim [batch_sz, input_sz]
			debug_assert!(pred.len() == (BATCH_SZ*dict_sz));
			
			// find max across each image, save output for printing, then set max = 1 and all other values to 0
			for img in 0..BATCH_SZ {
				// probabilstically sample
				let (max_ind, encoding_ind) = {
					let prob_val = rng.gen_f32b();
					let mut max_ind = 0;
					let mut val_sum = 0.;
					
					for ind in 0..dict_sz {
						let ind_use = img*dict_sz + ind;
						val_sum += pred[ind_use].exp();
						if prob_val < val_sum {
							max_ind = ind_use;
							break;
						}
					}
					(max_ind, max_ind - img*dict_sz)
				};
				// (max_ind indexes pred, encoding_ind indexes dict)
				
				if img == 0 {
					txt_output.push(if encoding_ind == (dict_sz-1) { // sequence termination character
						'\\'
					}else{
						dict[encoding_ind]
					});
				}
				
				// set vals
				for ind in 0..dict_sz {
					let ind_use = img*dict_sz + ind;
					pred[ind_use] = if ind_use != max_ind {
						0.
					}else{
						1.
					};
				}
			}
			pred
			
		// first input in sequence (nulls)
		}else{
			vec!{0.; t_stride}
		};
		
		model_cpu.layers[input_ind].set_output_seq(y_input, &vec!{1; BATCH_SZ});
		model_cpu.reset_fwd_cache_flags();
		model_cpu.forward(pred_ind);
	}
	
	println!("{}", txt_output.into_iter().collect::<String>());
	println!("{} secs", time.elapsed().as_millis() as f32 / 1000.);
	
	#[cfg(feature="profile")]
	write_prof();
}

