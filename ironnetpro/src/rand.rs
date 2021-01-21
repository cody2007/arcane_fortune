use std::num::Wrapping;
use std::time::{SystemTime, UNIX_EPOCH};
use super::{f32_to_f16, f16};

pub struct XorState {state: u32}

impl XorState {
	// https://en.wikipedia.org/w/index.php?title=Xorshift&oldid=910503315#Initialization
	pub fn init(seed: u64) -> XorState {
		let mut seed = Wrapping(seed);
		seed += Wrapping(0x9E3779B97f4A7C15);
		seed = (seed ^ (seed >> 30)) * Wrapping(0xBF58476D1CE4E5B9);
		seed = (seed ^ (seed >> 27)) * Wrapping(0x94D049BB133111EB);
		XorState {state: (seed ^ (seed >> 31)).0 as u32}
	}
	
	pub fn clock_init() -> XorState {
		let start = SystemTime::now();
		let elapsed = start.duration_since(UNIX_EPOCH).expect("The Unix epoch should be in the past. Check system clock settings.");
		XorState::init(elapsed.as_nanos() as u64)
	}
	
	// https://en.wikipedia.org/w/index.php?title=Xorshift&oldid=910503315#Example_implementation
	#[inline]
	pub fn gen(&mut self) -> u32 {
		self.state ^= self.state << 13;
		self.state ^= self.state >> 17;
		self.state ^= self.state << 5;
		self.state
	}

	// inclusive of lower, exclusive of upper
	#[inline]
	pub fn usize_range(&mut self, lower: usize, upper: usize) -> usize {
		((self.gen() as usize) % (upper-lower)) + lower
	}

	// inclusive of lower, exclusive of upper
	#[inline]
	pub fn isize_range(&mut self, lower: isize, upper: isize) -> isize {
		((self.gen() as isize) % (upper-lower)) + lower
	}
	
	// bounded between 0:1
	#[inline]
	pub fn gen_f32b(&mut self) -> f32 {
		(self.gen() as f32) / (std::u32::MAX as f32)
	}
	
	// normal distribution
	#[inline]
	pub fn gen_norm(&mut self, norm_scale: f32) -> f32 {
		const N_SAMPLES: usize = 12;
		
		let mut v = self.gen_f32b();
		for _ in 0..(N_SAMPLES-1) {
			v += self.gen_f32b();
		}
		
		norm_scale * (v - (N_SAMPLES/2) as f32)
	}
	
	pub fn gen_norm_vec(&mut self, len: usize, norm_scale: f32) -> Vec<f32> {
		let mut vals = vec!{0.; len};
		for v in vals.iter_mut() {
			*v = self.gen_norm(norm_scale);// + norm_scale/10.;
		}
		vals
	}
	
	pub fn gen_norm_vec_f16(&mut self, len: usize, norm_scale: f32) -> Vec<f16> {
		let mut vals = Vec::with_capacity(len);
		for _ in 0..len {
			vals.push(f32_to_f16(self.gen_norm(norm_scale)));
		}
		
		vals
	}
	
	// bounded between -norm_scale:norm_scale
	pub fn gen_vec(&mut self, len: usize, norm_scale: f32) -> Vec<f32> {
		let mut vals = vec!{0.; len};
		for v in vals.iter_mut() {
			*v = self.gen_f32b()*2.*norm_scale - norm_scale;
		}
		vals
	}
	
	pub fn gen_vec_usize(&mut self, len: usize, lower: usize, upper: usize) -> Vec<usize> {
		let mut vals = Vec::with_capacity(len);
		for _ in 0..len {
			vals.push(self.usize_range(lower, upper));
		}
		vals
	}
	
	#[inline]
	fn inds_internal(&mut self, len: usize, len_keep: usize) -> Box<[usize]> {
		debug_assert!(len > 0);
		debug_assert!(len_keep <= len);
		
		let mut seq: Box<[usize]> = (0..len).collect(); // inds not yet used
		let mut inds = vec![0; len_keep].into_boxed_slice();
		
		for (i, ind) in inds.iter_mut().enumerate() {
			let rand_sel = (self.gen() as usize) % (len-i);
			*ind = seq[rand_sel];
			debug_assert!(*ind < len);
			if rand_sel < (len-i-1) {
				seq[rand_sel] = seq[len-i-1];
			}
		}
		inds
	}
	
	// randomly shuffle 0..len
	#[inline]
	pub fn inds(&mut self, len: usize) -> Box<[usize]> {
		self.inds_internal(len, len)
	}
}

