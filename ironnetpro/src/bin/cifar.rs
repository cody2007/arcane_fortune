// cifar log linear loss, 10 categories
//	plotting: http://localhost:8888/notebooks/modeling_notebooks/training_ironnet_cifar.ipynb

extern crate ironnetpro;
use ironnetpro::*;
use std::os::raw::{c_int};
use std::time::Instant;

const MODEL_DIR: &str = "/tmp/cifar_test/";

const EPS: f32 = 1e-1;
const DATA_TYPE: cudnnDataType_t = cudnnDataType_t::CUDNN_DATA_FLOAT;
const NORM_SCALE: f32 = 1e-1;
const SAVE_FREQ: usize = 500;

const N_CATEGORIES: c_int = 10;

const BATCH_SZ: c_int = 128;
const INPUT_CHANNELS: c_int = 3;
const IMG_SZ: c_int = 32;

const CONV_PAD: c_int = 2;
const N_FILTERS: c_int = 64;
const FILTER_SZ: c_int = 3;
const CONV_STRIDE: c_int = 1;
	
const POOL_SZ: c_int = 3;
const POOL_STRIDE: c_int = 2;

const VALS_PER_IMG: usize = (INPUT_CHANNELS*IMG_SZ*IMG_SZ) as usize;
const VALS_PER_BATCH: usize = BATCH_SZ as usize * VALS_PER_IMG;

fn main() {
	///////////
	set_device(0);
	let mut rng = XorState::clock_init();
	
	/////////////////////////////////////////////////////////////////////////////////
	// training data pre-proc
	let (imgs, labels, n_batches) = {
		let label_inds = load_numpy::<u8>("/home/tapa/docs/datasets/ironnetpro/cifar/labels.npy");
		let imgs_u8 = load_numpy::<u8>("/home/tapa/docs/datasets/ironnetpro/cifar/imgs.npy");
		
		let n_batches = label_inds.len() / BATCH_SZ as usize;
		
		let mut imgs: Vec<f32> = Vec::with_capacity(imgs_u8.len());
		
		// subtract mean img
		{
			let n_total_imgs = label_inds.len();
			debug_assert!(imgs_u8.len() == (n_total_imgs*VALS_PER_IMG));
			
			let mut mean_img = vec!{0.; VALS_PER_IMG};
			for img_ind in 0..n_total_imgs {
				for (img_val, mean_img_val) in imgs_u8.iter().
									skip(img_ind*VALS_PER_IMG).
									take(VALS_PER_IMG).
									zip(mean_img.iter_mut()) {
					*mean_img_val += *img_val as f32 / n_total_imgs as f32;
				}
			}
			
			for img_ind in 0..n_total_imgs {
				for (img_val, mean_img_val) in imgs_u8.iter().
									skip(img_ind*VALS_PER_IMG).
									take(VALS_PER_IMG).
									zip(mean_img.iter()) {
					imgs.push((*img_val as f32 - *mean_img_val) / 255.);
				}
			}
		}
		
		// one hot
		let mut labels: Vec<f32> = Vec::with_capacity(label_inds.len()*N_CATEGORIES as usize);
		for label_ind in label_inds {
			for category in 0..N_CATEGORIES as u8 {
				labels.push(if category != label_ind {0.} else {1.});
			}
		}
		
		(imgs, labels, n_batches)
	};
	
	//////////////////////////////////////////////////////
	// model definition
	let mut model = {
		let mut model = Model::new(EPS / BATCH_SZ as f32, BATCH_SZ);
		
		let imgs_shape = TensorShape {
			n: BATCH_SZ,
			c: INPUT_CHANNELS,
			h: IMG_SZ,
			w: IMG_SZ
		};

		model.add_imgs(ImgParams {
					shape: imgs_shape,
					data_type: DATA_TYPE });
		
		///////// first conv layer
		model.add_conv(ConvParams {
					n_filters: N_FILTERS,
					filter_sz_h: 5,
					filter_sz_w: 5,
					pad_h: CONV_PAD,
					pad_w: CONV_PAD,
					stride: CONV_STRIDE,
					norm_scale: NORM_SCALE,
					data_type: DATA_TYPE }, &mut rng);
				
		//model.add_relu(DATA_TYPE);
		model.add_max_pooling(MaxPoolParams {
						pool_sz: POOL_SZ,
						pad_h: 0,
						pad_w: 0,
						stride: POOL_STRIDE,
						data_type: DATA_TYPE });
		
		//////////// conv layers
		for _ in 1..3 {
			model.add_conv(ConvParams {
						n_filters: N_FILTERS,
						filter_sz_h: FILTER_SZ,
						filter_sz_w: FILTER_SZ,
						pad_h: CONV_PAD,
						pad_w: CONV_PAD,
						stride: CONV_STRIDE,
						data_type: DATA_TYPE,
						norm_scale: NORM_SCALE }, &mut rng);
			
			//model.add_relu(DATA_TYPE);
			model.add_max_pooling(MaxPoolParams {
							pool_sz: POOL_SZ,
							pad_h: 0,
							pad_w: 0,
							stride: POOL_STRIDE,
							data_type: DATA_TYPE });
			
			println!("{}", model.layers.last().unwrap().y.tensor().shape.to_string());
		}
		
		///////////// fully connected layer
		//model.add_relu(DATA_TYPE);
		let y_shape = model.layers.last().unwrap().y.tensor().shape;
		model.add_conv(ConvParams {
					n_filters: N_CATEGORIES,
					filter_sz_h: y_shape.h,
					filter_sz_w: y_shape.w,
					pad_h: 0,
					pad_w: 0,
					stride: 1,
					norm_scale: NORM_SCALE,
					data_type: DATA_TYPE }, &mut rng);
		
		println!("{}", y_shape.to_string());
		
		model.add_softmax_cross_entropy_loss(DATA_TYPE);
		//model.add_correlation_loss(DATA_TYPE);
		
		model
	};
	
	let (labels_ind, pred_ind) = {
		let input_inds = model.input_layer_inds();
		let pred_inds = model.find_layer_inds("softmax_log");
		
		assert!(input_inds.len() == 2);
		assert!(pred_inds.len() == 1);
		
		(input_inds[1], pred_inds[0])
	};
	
	//////////////////////////////////////////////////////////
	// training loop
	{
		let mut time = Instant::now();
		let mut errs = Vec::new();
		for i in 0..1_000_000 {	
			let batch = i % n_batches as usize;
			
			let imgs_batch = &imgs[batch*VALS_PER_BATCH..(batch+1)*VALS_PER_BATCH].to_vec();
			let labels_batch = &labels[batch*(BATCH_SZ*N_CATEGORIES) as usize..(batch+1)*(BATCH_SZ*N_CATEGORIES) as usize].to_vec();
			
			model.layers[0].set_output(imgs_batch);
			model.layers[labels_ind].set_output(labels_batch);
			
			//model.grad_descent(model.layers.len()-1);
			model.rms_descent(model.layers.len()-1);
			if (i % SAVE_FREQ) == 0 {
				model.sv("/tmp/model");
				errs.push(model.layers.last().unwrap().y.ret()[0]);
				
				let pred = model.layers[pred_ind].y.ret();
				pred.sv("/tmp", "pred");
				labels_batch.sv("/tmp", "labels_batch");
				println!("{} err {} corr {} acc {} t {}", 
						i, errs.last().unwrap(), corr(&pred, &labels_batch), accuracy(&pred, &labels_batch, N_CATEGORIES as usize),
						time.elapsed().as_millis() as f32 / 1000.);
				for k in 0..10 {
					println!("{} {}", labels_batch[k], pred[k]);
				}
				errs.sv(MODEL_DIR, "errs");
				time = Instant::now();
			}
		}
	}
	
	/////////////////////////////////////////
	// test finite diff gradients
	{
		for layer in 1..model.layers.len() {
			println!("{}", layer);
			model.finite_diff_full_model_test(layer, model.layers.len()-1, &mut rng);
		}
	}
}

