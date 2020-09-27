// rustc testing/lstm_cpu_matching_gpu.rs -L ~/ironnetpro/target/release/ -L /usr/local/cuda/lib64; ./lstm_cpu_matching_gpu
extern crate ironnetpro;
use ironnetpro::*;
//use std::time::Instant;

const MODEL_DIR: &str = "/tmp/model";

const GPU_CARD: Cint = 0;
const BATCH_SZ: usize = 128;

fn main() {
	let mut rng = XorState::clock_init();
	
	/////////////
	set_device(GPU_CARD);
	
	let mut model_cpu = ModelCPU::load(MODEL_DIR, BATCH_SZ);
	let mut model = Model::load(MODEL_DIR);
	
	let (input_ind, labels_ind, pred_ind) = {
		let input_inds = model_cpu.input_layer_inds();
		let pred_inds = model_cpu.find_layer_inds("softmax_log");
		
		assert!(input_inds.len() == 2);
		assert!(pred_inds.len() == 1);
		
		(input_inds[0], input_inds[1], pred_inds[0])
	};
	
	// set inputs
	{
		let dict_sz = 80;
		let inputs = rng.gen_norm_vec(BATCH_SZ*dict_sz, 1e-1);
		model_cpu.layers[input_ind].set_output_seq(inputs.clone(), &vec!{1; BATCH_SZ});
		model.layers[input_ind].set_output_seq(&inputs, &vec!{1; BATCH_SZ});
	}
	
	for _ in 0..2 {
		model.reset_fwd_cache_flags();
		model.forward_inference(pred_ind);
		
		model_cpu.reset_fwd_cache_flags();
		model_cpu.forward(pred_ind);
		
		////////////////// outputs from same initial inputs
		for layer_ind in 1..=3 {
			let y_cpu = model_cpu.layers[layer_ind].y.mem();
			let y = model.layers[layer_ind].y.ret();
			
			println!("layer_ind {} {} y_cpu {} y {} {}", layer_ind, 
					model.layers[layer_ind].nm,
					y_cpu.len(), y.len(), y == *y_cpu);
			
			for i in 0..2 {
				println!("{} {}", y_cpu[i], y[i]);
			}
			println!("");
		}
		
		println!("---------");
	}
	
	let frac_matching = |a: &Vec<f32>, b: &Vec<f32>| {
		const EPS_TOL: f32 = 1e-3;
		let mut n_matching = 0;
		for (av, bv) in a.iter().zip(b.iter()) {
			n_matching += if (*av - *bv).abs() < EPS_TOL {1} else {0};
		}
		n_matching as f32 / a.len() as f32
	};
	
	model.zero_out_states();
	model.reset_fwd_cache_flags();
	model.forward_inference(pred_ind);
	
	////////////// outputs from same layer inputs
	for layer_ind in 1..=3 {
		/*for i in 0..3 {
			println!("x gpu {}", model.layers[layer_ind-1].y.ret()[i]);
		}*/
		
		// set input
		*model_cpu.layers[layer_ind-1].y.mem_mut() = model.layers[layer_ind-1].y.ret();
		
		// forward only layer_ind
		model_cpu.layers[layer_ind].run_fwd = false;
		model_cpu.zero_out_states();
		model_cpu.forward(layer_ind);
		
		let y_cpu = {
			let y_cpu = &model_cpu.layers[layer_ind].y;
			y_cpu.mem()[..y_cpu.ravel_time_shape().n_elements()].to_vec()
		};
		let y = model.layers[layer_ind].y.ret();
		
		println!("layer_ind {} {} y_cpu {} y {} frac_matching {}", layer_ind, 
				model.layers[layer_ind].nm,
				y_cpu.len(), y.len(), frac_matching(&y_cpu, &y));
		
		model_cpu.layers[layer_ind].y.ravel_time_shape().print();
		model.layers[layer_ind].y.ravel_time_shape().print();
		
		if layer_ind == 2 {
			sv_w_shape(&model.layers[layer_ind-1].y.ret(), "/tmp/", "x.npy", Some(model.layers[layer_ind-1].y.ravel_time_shape()));
			sv_w_shape(&y_cpu, "/tmp/", "y_cpu.npy", Some(model_cpu.layers[layer_ind].y.ravel_time_shape()));
			sv_w_shape(&y, "/tmp/", "y.npy", Some(model_cpu.layers[layer_ind].y.ravel_time_shape()));
		}
		
		for i in 0..2 {
			println!("{} {}", y_cpu[i], y[i]);
		}
		println!("");
	}
	
	////////// filters
	let f_cpu = if let InternalTypesCPU::Conv(internals) = &model_cpu.layer_internals[2] {
		sv_w_shape(&internals.filter.mem, "/tmp/", "filters.npy", Some(internals.filter.shape));
		internals.filter.mem.clone()
	}else{panic!();};
	
	let f = if let InternalTypes::Conv(internals) = &model.layers[2].internals {
		internals.filter.ret()
	}else{panic!();};
	
	println!("filters matching {}", frac_matching(&f_cpu, &f));
}

