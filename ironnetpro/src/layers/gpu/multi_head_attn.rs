#![allow(non_snake_case)]
use super::*;
use crate::layers::{MultiHeadAttnParams, QKVLayerParams,
		TransposeReshapeParams, FullyConnectedWBiasParams,
		FullyConnectedWBiasReluParams,
		QKPlusQPosMaskFutureTimesSoftmaxWMulVParams
		//QKPlusQPosMaskFutureTimesSoftmaxWParams
};

/*
	aka a transformer
	
	"Attention is all you need" Vaswani, et al. 2017
	
	explicit summation formulas (plus learned rel position values):
		 "Self-Attention with Relative Position Representations" Shaw, et al. 2018
	also see "Music Transformer: Genrating music with long-term structure" Cheng-Zhi Anna Huang, et al. 2018
		for a memory concise version of relative position. something like that is used here, but not exactly
		(a custom GPU kernel is used, which prevents the need for padding the position matrix)
*/

impl Model {
	pub fn add_multi_head_attn(&mut self, params: MultiHeadAttnParams, rng: &mut XorState) {
		let data_type = params.data_type;
	
		macro_rules! chk_shape{($n: expr, $c: expr, $h: expr, $w: expr) => {
			assert!(self.layers.last().unwrap().y.tensor().shape == 
				TensorShape {n: $n, c: $c, h: $h, w: $w});
		};}
	
		debug_assert!(self.layers.len() > 0);
		let x_layer_ind = self.layers.len() - 1;
		// X[batch, time, vec_in, 1]
		let x_shape = self.layers[x_layer_ind].y.tensor().shape;
		let batch_sz = x_shape.n;
		let n_time = x_shape.c;
		let vec_in = x_shape.h;
		debug_assert!(x_shape.w == 1);
		
		{ //////////// multi-head attn
			let mut add_QKV_layer = || {
				self.add_QKV_layer(x_layer_ind, QKVLayerParams {
					weight_initialization: WeightInitialization::XavierUniform(vec_in, vec_in),
					bias_initialization: WeightInitialization::XavierUniform(vec_in, 1),
					n_heads: params.n_heads,
					data_type
				}, rng);
				self.layers.len() - 1
			};
			
			let Q_layer_ind = add_QKV_layer();
			let K_layer_ind = add_QKV_layer();
			let V_layer_ind = add_QKV_layer();
			
			/*{ // softmax_w( mask_future_times(Q*K + Q*pos) )
				let vec_out = self.layers[Q_layer_ind].y.tensor().shape.w;
				// ^ Q[n_heads, batch_sz, n_time, vec_out]
				
				assert!(vec_out == (vec_in / params.n_heads));
				
				self.add_QK_plus_Qpos_mask_future_times_softmax_w(Q_layer_ind, K_layer_ind, 
					QKPlusQPosMaskFutureTimesSoftmaxWParams {
						weight_initialization: WeightInitialization::XavierUniform(n_time, vec_out),
						scale: (vec_out as f32).sqrt(),
						data_type
					}, rng);
				
				chk_shape!(params.n_heads, batch_sz, n_time, n_time);
			}
			
			self.add_mul_softmaxQK_and_V(V_layer_ind, MulSoftmaxQKAndVParams {data_type});*/
			
			{ // softmax_w( mask_future_times(Q*K + Q*pos) ) * V
				let vec_out = self.layers[Q_layer_ind].y.tensor().shape.w;
				// ^ Q[n_heads, batch_sz, n_time, vec_out]
				
				assert!(vec_out == (vec_in / params.n_heads));
				
				self.add_QK_plus_Qpos_mask_future_times_softmaxw_mul_V(Q_layer_ind, K_layer_ind, V_layer_ind,
					QKPlusQPosMaskFutureTimesSoftmaxWMulVParams {
						weight_initialization: WeightInitialization::XavierUniform(n_time, vec_out),
						scale: 1_f32/((vec_out as f32).sqrt()),
						data_type
					}, rng);
				chk_shape!(params.n_heads, batch_sz, n_time, vec_out);
			}
			
			{ // transpose, reshape
				// X[n_heads, batch_sz, n_time, vec_out] -> [batch_sz, n_time, n_heads*vec_out, 1]
				let x = self.layers[self.layers.len() - 1].y.tensor();
				
				let n_heads = x.shape.n;
				let batch_sz = x.shape.c;
				let n_time = x.shape.h;
				let vec_out = x.shape.w;
				debug_assert!(n_heads == params.n_heads);
				
				self.add_transpose_reshape(TransposeReshapeParams {
					fwd_dims: vec![1,2,0,3], // [0,1,2,3] -> [1,2,0,3]
					new_shape: TensorShape {
						n: batch_sz,
						c: n_time,
						h: n_heads*vec_out,
						w: 1
					},
					data_type
				});
			}
		}
		
		// bypass multi-head attn
		self.add_add(x_layer_ind, AddParams {alpha1: 1., alpha2: 1., data_type});
		
		// norm across dims 2 & 3 (x: [batch_sz, n_time, vec_in, 1])
		self.add_layer_norm(&vec![2,3], data_type);
		
		let before_feed_forward_ind = self.layers.len() - 1;
		
		{ /////////////// feed forward (2 fc layers)
			chk_shape!(batch_sz, n_time, vec_in, 1);
			self.add_fully_connected_w_bias_relu(FullyConnectedWBiasReluParams {
					vec_out_sz: params.feed_forward_sz,
					weight_initialization: WeightInitialization::XavierUniform(params.feed_forward_sz, vec_in),
					bias_initialization: WeightInitialization::XavierUniform(params.feed_forward_sz, 1),
					data_type
			}, rng);
			
			//self.add_relu(ActivationParams {data_type});
			
			chk_shape!(batch_sz, n_time, params.feed_forward_sz, 1);
			self.add_fully_connected_w_bias(FullyConnectedWBiasParams {
					vec_out_sz: vec_in,
					weight_initialization: WeightInitialization::XavierUniform(vec_in, params.feed_forward_sz),
					bias_initialization: WeightInitialization::XavierUniform(vec_in, 1),
					data_type
			}, rng);
			
			chk_shape!(batch_sz, n_time, vec_in, 1);
		}
		
		// bypass feed forward
		self.add_add(before_feed_forward_ind, AddParams {alpha1: 1., alpha2: 1., data_type});
		
		// norm across dims 2 & 3 (x: [batch_sz, n_time, vec_in, 1])
		chk_shape!(batch_sz, n_time, vec_in, 1);
		self.add_layer_norm(&vec![2,3], data_type);
	}
}

