/*extern crate ironnetpro;
use ironnetpro::*;
use std::os::raw::c_int;
use std::cmp::Ordering;
use std::time::Instant;

const GPU_CARD: c_int = 0;

const BATCH_SZ: usize = 12;

const DATA_TYPE: cudnnDataType_t = cudnnDataType_t::CUDNN_DATA_HALF;
//const DATA_TYPE: cudnnDataType_t = cudnnDataType_t::CUDNN_DATA_FLOAT;

const NORM_SCALE: f32 = 1e-1;

const N_HEADS: c_int = 8;
const VEC_IN: c_int = 16*N_HEADS;//64*N_HEADS;

fn main() {
	let mut rng = XorState::clock_init();
	
	set_device(GPU_CARD);
	
	let imgs_shape = TensorShape {
		n: BATCH_SZ as i32,
		c: 5,
		h: 7,
		w: 80
	};
	
	let mut model = Model::new(1e-5, BATCH_SZ as i32);
	
	model.add_imgs(ImgParams {
					shape: imgs_shape,
					data_type: DATA_TYPE });
	
	model.add_fully_connected(FullyConnectedParams {
			vec_out_sz: 1,
			norm_scale: NORM_SCALE,
			data_type: DATA_TYPE
	}, &mut rng);
	
	model.add_fully_connected(FullyConnectedParams {
			vec_out_sz: 1,
			norm_scale: NORM_SCALE,
			data_type: DATA_TYPE
	}, &mut rng);

	//model.add_softmax_cross_entropy_loss(cudnnDataType_t::CUDNN_DATA_FLOAT);
	//model.add_correlation_loss(DATA_TYPE);
	
	model.add_sum_reduce(SumReduceParams {sum_type: SumType::All, data_type: cudnnDataType_t::CUDNN_DATA_FLOAT});
	//model.add_sum_reduce(SumReduceParams {sum_type: SumType::All, data_type: DATA_TYPE});

	/*let (input_ind, labels_ind) = {
		let input_inds = model.input_layer_inds();
		assert!(input_inds.len() == 2);
		
		(input_inds[0], input_inds[1])
	};*/
	
	if DATA_TYPE == cudnnDataType_t::CUDNN_DATA_HALF {
		model.layers[0].set_output(&rng.gen_norm_vec_f16(imgs_shape.n_elements(), NORM_SCALE));
		//model.layers[labels_ind].set_output(&rng.gen_norm_vec_f16(model.layers[labels_ind].y.tensor().shape.n_elements(), NORM_SCALE));
		//model.layers[labels_ind].set_output(&rng.gen_norm_vec(model.layers[labels_ind].y.tensor().shape.n_elements(), NORM_SCALE));
	}else{
		model.layers[0].set_output(&rng.gen_norm_vec(imgs_shape.n_elements(), NORM_SCALE));
		//model.layers[labels_ind].set_output(&rng.gen_norm_vec(model.layers[labels_ind].y.tensor().shape.n_elements(), NORM_SCALE));
	}
	
	let mut time = Instant::now();
	
	for step in 0..15_000 {
		//println!("{}", step);
		model.adam_descent(model.layers.len()-1);
		if step % 200 == 0 {
			println!("{} {}", step, model.layers.last().unwrap().y.ret()[0]);
			/*for (layer_ind, layer) in model.layers.iter().enumerate() {
				println!("  {} {}", layer_ind, layer.y.tensor().ret().iter().max_by(|a,b| a.partial_cmp(b).unwrap_or(Ordering::Equal)).unwrap());
			}*/
			//println!("");
		}
	}
	
	println!("{}", time.elapsed().as_millis() as f32 / 1000.);
	/*if let InternalTypes::FullyConnected(internals) = &model.layers[1].internals {
		println!("  {}", internals.W.ret().iter().max_by(|a,b| a.partial_cmp(b).unwrap_or(Ordering::Equal)).unwrap());
	}*/
}
*/
fn main() {}
