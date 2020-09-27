use std::num::Wrapping;
use std::time::{SystemTime, UNIX_EPOCH};
use crate::disp_lib::endwin;
use crate::units::UnitTemplate;
use crate::buildings::BldgTemplate;
use crate::resources::ResourceTemplate;
use crate::doctrine::DoctrineTemplate;
use crate::saving::*;

#[derive(Clone, PartialEq)]
pub struct XorState {state: u32}

impl_saving!{XorState {state}}

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
		#[cfg(feature="fixed_seed")]
		return XorState::init(666);
		
		#[cfg(not(feature="fixed_seed"))]
		{
			let start = SystemTime::now();
			let elapsed = start.duration_since(UNIX_EPOCH).expect("The Unix epoch should be in the past. Check system clock settings.");
			XorState::init(elapsed.as_nanos() as u64)
		}
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
	
	pub fn usize_range_vec(&mut self, lower: usize, upper: usize, len: usize) -> Vec<usize> {
		let mut vals = vec!{0; len};
		for v in vals.iter_mut() {
			*v = self.usize_range(lower, upper);
			debug_assertq!(*v >= lower && *v < upper);
		}
		vals
	}
	
	// bounded between 0:1
	#[inline]
	pub fn gen_f32b(&mut self) -> f32 {
		(self.gen() as f32) / (std::u32::MAX as f32)
	}
	
	// bounded between 0:1
	pub fn f32b_vec(&mut self, len: usize) -> Vec<f32> {
		let mut vals = vec!{0.; len};
		for v in vals.iter_mut() {
			*v = self.gen_f32b();
			debug_assertq!(*v >= 0. && *v < 1.);
		}
		vals
	}
	
	// normal distribution
	#[inline]
	pub fn gen_norm(&mut self) -> f32 {
		const N_SAMPLES: usize = 12;
		
		let mut v = self.gen_f32b();
		for _ in 0..(N_SAMPLES-1) {
			v += self.gen_f32b();
		}
		
		v - (N_SAMPLES/2) as f32
	}
	
	// shuffle all elements in-place. could be more efficient?
	pub fn shuffle<T: Copy>(&mut self, vals: &mut Vec<T>) {
		let sz = vals.len();
		debug_assertq!(sz > 0);
		
		for i in 0..sz {
			let j = (self.gen() as usize) % sz;
			let val_cp = vals[i];
			vals[i] = vals[j];
			vals[j] = val_cp;
		}
	}
	
	#[inline]
	fn inds_internal(&mut self, len: usize, len_keep: usize) -> Box<[usize]> {
		debug_assertq!(len > 0);
		debug_assertq!(len_keep <= len);
		
		let mut seq: Box<[usize]> = (0..len).collect(); // inds not yet used
		let mut inds = vec![0; len_keep].into_boxed_slice();
		
		for (i, ind) in inds.iter_mut().enumerate() {
			let rand_sel = (self.gen() as usize) % (len-i);
			*ind = seq[rand_sel];
			debug_assertq!(*ind < len);
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

	// return max_len values in range 0..len, all unique
	pub fn inds_max(&mut self, len: usize, max_len: usize) -> Box<[usize]> {
		if len <= max_len {
			(0..len).collect()
		}else{
			//self.inds(len)[0..max_len].into()
			self.inds_internal(len, max_len) // ???
		}
	}
}

