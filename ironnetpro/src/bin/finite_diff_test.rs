extern crate ironnetpro;
use ironnetpro::*;
use std::os::raw::{c_int};
//use std::time::Instant;

const GPU_CARD: c_int = 0;
const DATA_TYPE: cudnnDataType_t = cudnnDataType_t::CUDNN_DATA_FLOAT;
const NORM_SCALE: f32 = 1e-1;

const BATCH_SZ: i32 = 2;
const MAX_SEQ_LEN: i32 = 2;
const VEC_IN: i32 = 4;

const N_HEADS: i32 = 2;
const FEED_FORWARD_SZ: i32 = 5;

fn main() {
	let mut rng = XorState::clock_init();
	let dict_sz = 30;
	
	set_device(GPU_CARD);
	
	let imgs_shape = TensorShape {
		n: BATCH_SZ as i32,
		c: MAX_SEQ_LEN as i32,
		h: dict_sz as i32,
		w: 1
	};

	let mut model = {
		let mut model = Model::new(0., BATCH_SZ as i32);
		
		model.add_imgs(ImgParams {
					shape: imgs_shape,
					data_type: DATA_TYPE });
		
		model.add_fully_connected_w_bias(FullyConnectedWBiasParams {
				vec_out_sz: VEC_IN,
				weight_initialization: WeightInitialization::NormScale(NORM_SCALE),
				bias_initialization: WeightInitialization::NormScale(NORM_SCALE),
				data_type: DATA_TYPE
		}, &mut rng);
		//model.add_bias_channels(BiasParams {norm_scale: NORM_SCALE, data_type: DATA_TYPE}, &mut rng);
		
		model.add_multi_head_attn(MultiHeadAttnParams {
				n_heads: N_HEADS,
				feed_forward_sz: FEED_FORWARD_SZ,
				data_type: DATA_TYPE
			}, &mut rng);
		
		model
	};
	
	let inputs = rng.gen_norm_vec(imgs_shape.n_elements(), NORM_SCALE);
	model.layers[0].set_output(&inputs);
	
	model.reset_fwd_cache_flags();
	model.forward(model.layers.len() - 1);
	
	for layer_ind in 1..model.layers.len() {
		model.finite_diff_test(layer_ind);
		//let layer_ind = 5;
		//model.finite_diff_weights_test(layer_ind);
	}
}

