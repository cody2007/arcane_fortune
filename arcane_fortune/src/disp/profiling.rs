use std::cmp::max;
use std::time::Instant;

use crate::saving::*;
use crate::buildings::BldgTemplate;
use crate::units::UnitTemplate;
use crate::resources::ResourceTemplate;
use crate::doctrine::DoctrineTemplate;

const FRAME_DUR_SZ: usize = 50;

#[derive(PartialEq, Clone)]
pub struct FrameStats {
	pub init: bool,
	pub dur_mean: f32,
	dur_ind: usize,
	dur_buf: Vec<f32>
}

impl_saving!{FrameStats {init, dur_mean, dur_ind, dur_buf}}

impl FrameStats {
	pub fn init() -> Self {
		Self {init: true, dur_mean: 10., dur_ind: 0, dur_buf: vec!{0.; FRAME_DUR_SZ}}
	}
	
	pub fn update(&mut self, frame_start: Instant) {
		self.dur_mean -= self.dur_buf[self.dur_ind] / (FRAME_DUR_SZ as f32);
		self.dur_buf[self.dur_ind] = (frame_start.elapsed().as_micros() as f32) / 1e3;
		self.dur_mean += self.dur_buf[self.dur_ind] / (FRAME_DUR_SZ as f32);
		if self.init {
			for i in 1..FRAME_DUR_SZ {
				self.dur_buf[i] = self.dur_buf[0];
			}
			self.init = false;
			self.dur_mean = self.dur_buf[0];
		}
		
		self.dur_ind = (self.dur_ind + 1) % FRAME_DUR_SZ;
	}
	
	pub fn days_per_frame(&self) -> usize {
		const MIN_FRAME_RATE: f32 = 40.;
		const MAX_FRAME_TIME: f32 = 1000./MIN_FRAME_RATE;
		max(1, (MAX_FRAME_TIME/self.dur_mean).round() as usize)
	}
}

impl Default for FrameStats {
	fn default() -> Self {
		Self {init: false, dur_mean: 0., dur_ind: 0, dur_buf: Vec::new() }}}
