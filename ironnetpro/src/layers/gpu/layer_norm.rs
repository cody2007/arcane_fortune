use super::*;
use crate::layers::{SumReduceParams, SumType, MulParams, ScaleParams, ElementwiseAffineParams};

impl Model {
	// (x - x.mean()) / x.std() for dims_norm (ex, [2,3])
	pub fn add_layer_norm(&mut self, dims_norm: &Vec<usize>, data_type: cudnnDataType_t) {
		debug_assert!(self.layers.len() > 0);
		
		let norm_input_ind = self.layers.len() - 1;
		
		// sum across feed_forward_sz (x: [batch_sz, n_time, params.feed_forward_sz, 1])
		let x_shape = self.layers[norm_input_ind].y.tensor().shape;
		//debug_assert!(x_shape.n_elements() == (batch_sz*n_time*vec_in) as usize,
		//		"{} {} {}  {}", batch_sz, n_time, vec_in, x_shape.to_string());
		//println!("{}", x_shape.to_string());
		
		let x_szs = vec![x_shape.n, x_shape.c, x_shape.h, x_shape.w];
		
		let mut n_reduce_elements = 1;
		for dim in dims_norm.iter() {
			n_reduce_elements *= x_szs[*dim];
		}
		
		let sum_reduce_params = SumReduceParams {
				sum_type: SumType::Axes(dims_norm.clone()),
				data_type
		};
		
		self.add_sum_reduce(sum_reduce_params.clone());
		
		// norm_input - norm_input_mean
		self.add_add(norm_input_ind, AddParams {
				alpha1: -1./(n_reduce_elements as f32),
				alpha2: 1.,
				data_type
		});
		let norm_input_zmean_ind = self.layers.len() - 1;
		
		// sum((norm_input - norm_input_mean)^2)
		self.add_pow(PowParams {alpha: 2., data_type});
		self.add_sum_reduce(sum_reduce_params.clone());
		self.add_scale(ScaleParams {alpha: 1./n_reduce_elements as f32, data_type});
			// ^ (1/H) * sum(...)
		
		// 1/sqrt(sum_sqs)
		self.add_pow(PowParams {alpha: -0.5, data_type});
		
		////// norm_input_zmean / sqrt(sum(norm_input_zmean^2))
		//println!("before mul {}", self.layers[self.layers.len() - 1].y.tensor().shape.to_string());
		//println!("... {}", self.layers[norm_input_zmean_ind].y.tensor().shape.to_string());
		
		self.add_mul(norm_input_zmean_ind, MulParams {data_type});
		
		self.add_elementwise_affine(ElementwiseAffineParams {data_type, dims: dims_norm.clone()});
	}
}

