// this is a terminal (input) layer
use std::os::raw::c_int;
use super::*;
use crate::layers::TimeSeriesParams;
use crate::saving::{KeyPair, find_req_key_parse};

pub struct TimeSeriesInternalsCPU {}

impl RunCPU for TimeSeriesInternalsCPU {
	fn forward(&mut self, _layer_ind: usize, _: &mut Vec<LayerCPU>) {}
}

impl ModelCPU {
	pub fn add_time_series(&mut self, params: TimeSeriesParams) {
		let max_seq_len = params.max_seq_len;
		let batch_sz = params.batch_sz;
		let vec_sz = params.vec_sz;
		
		let seq_len_array = vec!{max_seq_len as c_int; batch_sz as usize};
			// ^ "An integer array with batchSize number of elements.
			// 	Describes the length (number of time-steps) of each sequence." -- DA-09702-001_v7.6.5 | 330
		
		self.new_layer_time_series( 
				Vec::new(), // no inputs
				InternalTypesCPU::TimeSeries(TimeSeriesInternalsCPU {}),
				RNNDataCPU::new(max_seq_len, batch_sz, vec_sz, &seq_len_array),
				String::from("time_series")
		);
	}
	
	pub fn load_time_series(&mut self, layer_keys: &Vec<KeyPair>) {
		let data_type = find_req_key_parse("data_type", &layer_keys);
		debug_assert!(data_type == cudnnDataType_t::CUDNN_DATA_FLOAT);
		
		self.add_time_series(TimeSeriesParams {
			max_seq_len: find_req_key_parse("max_seq_len", layer_keys),
			batch_sz: self.batch_sz,
			vec_sz: find_req_key_parse("vec_sz", layer_keys),
			data_type
		});
	}
}

