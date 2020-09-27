// this is a terminal (input) layer
use std::os::raw::c_int;
use super::*;
use crate::layers::TimeSeriesParams;
use crate::saving::{KeyPair, find_req_key_parse};

pub struct TimeSeriesInternals {params: TimeSeriesParams}

impl Run for TimeSeriesInternals {
	fn forward(&self, _: &Layer, _: &Model) {}
	fn backward(&self, _: &Layer, _: &Model) {}
	
	fn sv_arch(&self, txt: &mut String) {
		txt.push_str(&format!("\tmax_seq_len: {}\n", self.params.max_seq_len));
		txt.push_str(&format!("\tbatch_sz: {}\n", self.params.batch_sz));
		txt.push_str(&format!("\tvec_sz: {}\n", self.params.vec_sz));
		txt.push_str(&format!("\tdata_type: {}\n", self.params.data_type));
	}
}

impl Model {
	pub fn add_time_series(&mut self, params: TimeSeriesParams) {
		let data_type = params.data_type;
		let max_seq_len = params.max_seq_len;
		let batch_sz = params.batch_sz;
		let vec_sz = params.vec_sz;
		
		let seq_len_array = vec!{max_seq_len as c_int; batch_sz as usize};
			// ^ "An integer array with batchSize number of elements.
			// 	Describes the length (number of time-steps) of each sequence." -- DA-09702-001_v7.6.5 | 330
		
		self.layers.push( Layer::new_time_series(
						Vec::new(), // no inputs
						InternalTypes::TimeSeries(TimeSeriesInternals {params}),
						RNNData::new(data_type, max_seq_len, batch_sz, vec_sz, &seq_len_array),
						String::from("time_series"),
						data_type
		));
	}
	
	pub fn load_time_series(&mut self, layer_keys: &Vec<KeyPair>) {
		self.add_time_series(TimeSeriesParams {
			max_seq_len: find_req_key_parse("max_seq_len", layer_keys),
			//batch_sz: self.batch_sz,
			batch_sz: find_req_key_parse("batch_sz", layer_keys),
			vec_sz: find_req_key_parse("vec_sz", layer_keys),
			data_type: find_req_key_parse("data_type", layer_keys)
		});
	}
}

