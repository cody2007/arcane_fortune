use super::*;

impl Default for cudnnConvolutionFwdAlgo_t {
	fn default() -> Self {
		Self::CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM
	}
}

impl Default for cudnnConvolutionBwdDataAlgo_t {
	fn default() -> Self {
		Self::CUDNN_CONVOLUTION_BWD_DATA_ALGO_0
	}
}

impl Default for cudnnConvolutionBwdFilterAlgo_t {
	fn default() -> Self {
		Self::CUDNN_CONVOLUTION_BWD_FILTER_ALGO_0
	}
}

pub struct WorkspaceConvolutionFwd {
	pub alg: cudnnConvolutionFwdAlgo_t,
	pub mem: gpuMem
}

pub struct WorkspaceConvolutionBwdData {
	pub alg: cudnnConvolutionBwdDataAlgo_t,
	pub mem: gpuMem
}

pub struct WorkspaceConvolutionBwdFilter {
	pub alg: cudnnConvolutionBwdFilterAlgo_t,
	pub mem: gpuMem
}

impl Handle {
	// max_sz not used when pref is ..._PREFER_FASTEST
	pub fn allocate_workspace_convolution_fwd_pref(&self,
			x: &Output,
			filter: &Filter,
			conv_desc: &ConvolutionDescriptor,
			y: &Output,
			pref: cudnnConvolutionFwdPreference_t,
			max_sz: size_t) -> WorkspaceConvolutionFwd {
		let mut alg = cudnnConvolutionFwdAlgo_t::default();
		
		unsafe {cudnnGetConvolutionForwardAlgorithm(self.cudnn_val,
				x.ravel_time_tensor_desc(), filter.desc.val, conv_desc.val,
				y.ravel_time_tensor_desc(),
				pref, max_sz, &mut alg)}.chk_err();
		
		let mut workspace_sz = 0;
		unsafe {cudnnGetConvolutionForwardWorkspaceSize(self.cudnn_val,
				x.ravel_time_tensor_desc(), filter.desc.val, conv_desc.val,
				y.ravel_time_tensor_desc(),
				alg, &mut workspace_sz)}.chk_err();
	
		WorkspaceConvolutionFwd {
			alg,
			mem: gpuMem::new_bytes(workspace_sz)
		}
	}
	
	// max_sz not used when pref is ..._PREFER_FASTEST
	pub fn allocate_workspace_convolution_bwd_data_pref(&self,
			filter: &Filter,
			dy: &Output,
			conv_desc: &ConvolutionDescriptor,
			dx: &Output,
			pref: cudnnConvolutionBwdDataPreference_t,
			max_sz: size_t) -> WorkspaceConvolutionBwdData {
		let mut alg = cudnnConvolutionBwdDataAlgo_t::default();
		
		unsafe {cudnnGetConvolutionBackwardDataAlgorithm(self.cudnn_val,
				filter.desc.val, dy.ravel_time_tensor_desc(),
				conv_desc.val,
				dx.ravel_time_tensor_desc(),
				pref, max_sz, &mut alg)}.chk_err();
		
		let mut workspace_sz = 0;
		unsafe {cudnnGetConvolutionBackwardDataWorkspaceSize(self.cudnn_val,
				filter.desc.val, dy.ravel_time_tensor_desc(),
				conv_desc.val,
				dx.ravel_time_tensor_desc(),
				alg, &mut workspace_sz)}.chk_err();
		
		WorkspaceConvolutionBwdData {
			alg,
			mem: gpuMem::new_bytes(workspace_sz)
		}
	}
	
	// max_sz not used when pref is ..._PREFER_FASTEST
	pub fn allocate_workspace_convolution_bwd_filter_pref(&self,
			x: &Output,
			dy: &Output,
			conv_desc: &ConvolutionDescriptor,
			dfilter: &Filter,
			pref: cudnnConvolutionBwdFilterPreference_t,
			max_sz: size_t) -> WorkspaceConvolutionBwdFilter {
		let mut alg = cudnnConvolutionBwdFilterAlgo_t::default();
		
		unsafe {cudnnGetConvolutionBackwardFilterAlgorithm(self.cudnn_val,
				x.ravel_time_tensor_desc(), dy.ravel_time_tensor_desc(), 
				conv_desc.val, dfilter.desc.val, pref, max_sz, &mut alg)}.chk_err();
		
		let mut workspace_sz = 0;
		unsafe {cudnnGetConvolutionBackwardFilterWorkspaceSize(self.cudnn_val,
				x.ravel_time_tensor_desc(), dy.ravel_time_tensor_desc(),
			conv_desc.val, dfilter.desc.val, alg, &mut workspace_sz)}.chk_err();
		
		WorkspaceConvolutionBwdFilter {
			alg,
			mem: gpuMem::new_bytes(workspace_sz)
		}
	}
}

