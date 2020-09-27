#[cfg(not(feature="cpu_only"))]
use super::{f16, f32_to_f16};

// common forward/backward functions
macro_rules! impl_get_req_layers{() => {
	pub fn get_req_layers_recursive(&self, layer_ind: usize, req_layers: &mut Vec<usize>) {
		for req_layer in self.layers[layer_ind].x_layers.iter() {
			if req_layers.contains(req_layer) {continue;} // already added
			self.get_req_layers_recursive(*req_layer, req_layers);
		}
		req_layers.push(layer_ind);
	}
	
	pub fn get_req_layers(&self, layer_ind: usize) -> Vec<usize> {
		let mut req_layers = Vec::with_capacity(self.layers.len());
		self.get_req_layers_recursive(layer_ind, &mut req_layers);
		req_layers
	}
	
	pub fn reset_fwd_cache_flags(&mut self) {
		for layer in self.layers.iter_mut() {
			layer.run_fwd = false;
		}
	}
	
	pub fn find_layer_inds(&self, nm: &str) -> Vec<usize> {
		let mut inds = Vec::new();
		for (ind, layer) in self.layers.iter().enumerate() {
			if layer.nm == nm {
				inds.push(ind);
			}
		}
		inds
	}
}}

#[cfg(not(feature="cpu_only"))]
pub mod gpu;
#[cfg(not(feature="cpu_only"))]
pub use gpu::*;

pub mod cpu; pub use cpu::*;

