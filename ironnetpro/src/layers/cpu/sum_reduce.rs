use super::*;
use crate::saving::{KeyPair, find_req_key_parse};
use crate::layers::{SumType, SumReduceParams};

pub struct SumReduceInternalsCPU {params: SumReduceParams}

impl RunCPU for SumReduceInternalsCPU {
	fn forward(&mut self, layer_ind: usize, layers: &mut Vec<LayerCPU>) {
		let layer = &layers[layer_ind];
		debug_assert!(layer.x_layers.len() == 1);
		
		let x = &layers[layer.x_layers[0]].y; // output of input layer is the input for this layer
		
		match self.params.sum_type {
			SumType::All => {
				*layers[layer_ind].y.mem_mut() = vec![
						(*x.mem()).iter().sum()
				];
			}
			SumType::Axes(_) => {panic!("not yet implemented sum_reduce SumType::Axes");}
		}
	}
}

impl ModelCPU {
	pub fn add_sum_reduce(&mut self, params: SumReduceParams) {
		debug_assert!(self.layers.len() > 0);
		
		let layer_prev_ind = self.layers.len() - 1;
		
		let x = self.layers[layer_prev_ind].y.tensor(); // (input to this layer is the output of the previous layer)
		
		let output_shape = match &params.sum_type {
			SumType::All => {TensorShape {n: 1, c: 1, h: 1, w: 1}}
			SumType::Axes(axes) => {
				let mut output_shape = x.shape;
				for axis in axes {
					match axis {
						0 => {output_shape.n = 1;}
						1 => {output_shape.c = 1;}
						2 => {output_shape.h = 1;}
						3 => {output_shape.w = 1;}
						_ => {panic!("unknown dimension");}
					}
				}
				output_shape
			}
		};
		
		let y = TensorCPU::new(output_shape);
		
		self.new_layer(
			vec![layer_prev_ind],
			InternalTypesCPU::SumReduce(SumReduceInternalsCPU {params}),
			y,
			String::from("sum_reduce")
		);
	}
	
	pub fn load_sum_reduce(&mut self, layer_keys: &Vec<KeyPair>) {
		let data_type = find_req_key_parse("data_type", &layer_keys);
		debug_assert!(data_type == cudnnDataType_t::CUDNN_DATA_FLOAT);

		self.add_sum_reduce(SumReduceParams {
			sum_type: find_req_key_parse("sum_type", layer_keys),
			data_type
		});
	}
}

