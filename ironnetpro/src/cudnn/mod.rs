#![allow(non_snake_case)]
#![allow(non_camel_case_types)]

use std::env;
use std::ptr::null_mut;
use std::ffi::c_void;
use std::mem::size_of;
use std::os::raw::{c_int, c_double, c_float, c_ulonglong};
use super::{f16, f16_to_f32};

pub mod raw; pub use raw::*;
pub mod workspace_utils; pub use workspace_utils::*;

use crate::data_wrappers::*;
pub use crate::layers::Output;

impl cudnnStatus_t {
	pub fn chk_err(&self) {
		match self {
			cudnnStatus_t::CUDNN_STATUS_SUCCESS => {}
			cudnnStatus_t::CUDNN_STATUS_NOT_INITIALIZED => {panic!("CUDNN_STATUS_NOT_INITIALIZED");}
			cudnnStatus_t::CUDNN_STATUS_ALLOC_FAILED => {panic!("CUDNN_STATUS_ALLOC_FAILED");}
			cudnnStatus_t::CUDNN_STATUS_BAD_PARAM => {panic!("CUDNN_STATUS_BAD_PARAM");}
			cudnnStatus_t::CUDNN_STATUS_INTERNAL_ERROR => {panic!("CUDNN_STATUS_INTERNAL_ERROR");}
			cudnnStatus_t::CUDNN_STATUS_INVALID_VALUE => {panic!("CUDNN_STATUS_INVALID_VALUE");}
			cudnnStatus_t::CUDNN_STATUS_ARCH_MISMATCH => {panic!("CUDNN_STATUS_ARCH_MISMATCH");}
			cudnnStatus_t::CUDNN_STATUS_MAPPING_ERROR => {panic!("CUDNN_STATUS_MAPPING_ERROR");}
			cudnnStatus_t::CUDNN_STATUS_EXECUTION_FAILED => {panic!("CUDNN_STATUS_EXECUTION_FAILED");}
			cudnnStatus_t::CUDNN_STATUS_NOT_SUPPORTED => {panic!("CUDNN_STATUS_NOT_SUPPORTED");}
			cudnnStatus_t::CUDNN_STATUS_LICENSE_ERROR => {panic!("CUDNN_STATUS_LICENSE_ERROR");}
			cudnnStatus_t::CUDNN_STATUS_RUNTIME_PREREQUISITE_MISSING => {panic!("CUDNN_STATUS_RUNTIME_PREREQUISITE_MISSING");}
			cudnnStatus_t::CUDNN_STATUS_RUNTIME_IN_PROGRESS => {panic!("CUDNN_STATUS_RUNTIME_IN_PROGRESS");}
			cudnnStatus_t::CUDNN_STATUS_RUNTIME_FP_OVERFLOW => {panic!("CUDNN_STATUS_RUNTIME_FP_OVERFLOW");}
		}
	}
}

impl cublasStatus_t {
	pub fn chk_err(&self) {
		match self {
			cublasStatus_t::CUBLAS_STATUS_SUCCESS => {}
			cublasStatus_t::CUBLAS_STATUS_NOT_INITIALIZED => {panic!("CUBLAS_STATUS_NOT_INITIALIZED");}
			cublasStatus_t::CUBLAS_STATUS_ALLOC_FAILED => {panic!("CUBLAS_STATUS_ALLOC_FAILED");}
			cublasStatus_t::CUBLAS_STATUS_INVALID_VALUE => {panic!("CUBLAS_STATUS_INVALID_VALUE");}
			cublasStatus_t::CUBLAS_STATUS_ARCH_MISMATCH => {panic!("CUBLAS_STATUS_ARCH_MISMATCH");}
			cublasStatus_t::CUBLAS_STATUS_MAPPING_ERROR => {panic!("CUBLAS_STATUS_MAPPING_ERROR");}
			cublasStatus_t::CUBLAS_STATUS_EXECUTION_FAILED => {panic!("CUBLAS_STATUS_EXECUTION_FAILED");}
			cublasStatus_t::CUBLAS_STATUS_INTERNAL_ERROR => {panic!("CUBLAS_STATUS_INTERNAL_ERROR");}
			cublasStatus_t::CUBLAS_STATUS_NOT_SUPPORTED => {panic!("CUBLAS_STATUS_NOT_SUPPORTED");}
			cublasStatus_t::CUBLAS_STATUS_LICENSE_ERROR => {panic!("CUBLAS_STATUS_LICENSE_ERROR");}
		}
	}
}

impl cudaError_t {
	pub fn chk_err(&self) {
		assert!(*self == cudaError_t::cudaSuccess, "cudaError: {}", *self as usize);
		/*match self {
			cudaSuccess => {panic!("
			cudaErrorInvalidValue => {panic!("
			cudaErrorMemoryAllocation => {panic!("
			cudaErrorInitializationError => {panic!("
			cudaErrorCudartUnloading => {panic!(",
			cudaErrorProfilerDisabled => {panic!("
			cudaErrorProfilerNotInitialized => {panic!("
			cudaErrorProfilerAlreadyStarted => {panic!("
			cudaErrorProfilerAlreadyStopped => {panic!("
			cudaErrorInvalidConfiguration => {panic!("
			cudaErrorInvalidPitchValue => {panic!("
			cudaErrorInvalidSymbol => {panic!("
			cudaErrorInvalidHostPointer => {panic!("
			cudaErrorInvalidDevicePointer => {panic!("
			cudaErrorInvalidTexture => {panic!("
			cudaErrorInvalidTextureBinding => {panic!("
			cudaErrorInvalidChannelDescriptor => {panic!("
			cudaErrorInvalidMemcpyDirection => {panic!("
			cudaErrorAddressOfConstant => {panic!("
			cudaErrorTextureFetchFailed => {panic!("
			cudaErrorTextureNotBound => {panic!("
			cudaErrorSynchronizationError => {panic!("
			cudaErrorInvalidFilterSetting => {panic!("
			cudaErrorInvalidNormSetting => {panic!("
			cudaErrorMixedDeviceExecution => {panic!("
			cudaErrorNotYetImplemented => {panic!("
			cudaErrorMemoryValueTooLarge => {panic!("
			cudaErrorInsufficientDriver => {panic!("
			cudaErrorInvalidSurface => {panic!("
			cudaErrorDuplicateVariableName => {panic!("
			cudaErrorDuplicateTextureName => {panic!("
			cudaErrorDuplicateSurfaceName => {panic!("
			cudaErrorDevicesUnavailable => {panic!("
			cudaErrorIncompatibleDriverContext => {panic!("
			cudaErrorMissingConfiguration => {panic!("
			cudaErrorPriorLaunchFailure => {panic!("
			cudaErrorLaunchMaxDepthExceeded => {panic!("
			cudaErrorLaunchFileScopedTex => {panic!("
			cudaErrorLaunchFileScopedSurf => {panic!("
			cudaErrorSyncDepthExceeded => {panic!("
			cudaErrorLaunchPendingCountExceeded => {panic!("
			cudaErrorInvalidDeviceFunction => {panic!("
			cudaErrorNoDevice => {panic!("
			cudaErrorInvalidDevice => {panic!("
			cudaErrorStartupFailure => {panic!("
			cudaErrorInvalidKernelImage => {panic!("
			cudaErrorDeviceUninitialized => {panic!("
			cudaErrorMapBufferObjectFailed => {panic!("
			cudaErrorUnmapBufferObjectFailed => {panic!("
			cudaErrorArrayIsMapped = 207,
			cudaErrorAlreadyMapped = 208,
			cudaErrorNoKernelImageForDevice = 209,
			cudaErrorAlreadyAcquired = 210,
			cudaErrorNotMapped = 211,
			cudaErrorNotMappedAsArray = 212,
			cudaErrorNotMappedAsPointer = 213,
			cudaErrorECCUncorrectable = 214,
			cudaErrorUnsupportedLimit = 215,
			cudaErrorDeviceAlreadyInUse = 216,
			cudaErrorPeerAccessUnsupported = 217,
			cudaErrorInvalidPtx = 218,
			cudaErrorInvalidGraphicsContext = 219,
			cudaErrorNvlinkUncorrectable = 220,
			cudaErrorJitCompilerNotFound = 221,
			cudaErrorInvalidSource = 300,
			cudaErrorFileNotFound = 301,
			cudaErrorSharedObjectSymbolNotFound = 302,
			cudaErrorSharedObjectInitFailed = 303,
			cudaErrorOperatingSystem = 304,
			cudaErrorInvalidResourceHandle = 400,
			cudaErrorIllegalState = 401,
			cudaErrorSymbolNotFound = 500,
			cudaErrorNotReady = 600,
			cudaErrorIllegalAddress = 700,
			cudaErrorLaunchOutOfResources = 701,
			cudaErrorLaunchTimeout = 702,
			cudaErrorLaunchIncompatibleTexturing = 703,
			cudaErrorPeerAccessAlreadyEnabled => {panic!("
			cudaErrorPeerAccessNotEnabled => {panic!("
			cudaErrorSetOnActiveProcess => {panic!("
			cudaErrorContextIsDestroyed => {panic!("
			cudaErrorAssert => {panic!("
			cudaErrorTooManyPeers => {panic!("
			cudaErrorHostMemoryAlreadyRegistered => {panic!("
			cudaErrorHostMemoryNotRegistered => {panic!("
			cudaErrorHardwareStackError => {panic!("
			cudaErrorIllegalInstruction => {panic!("
			cudaErrorMisalignedAddress => {panic!("
			cudaErrorInvalidAddressSpace => {panic!("
			cudaErrorInvalidPc => {panic!("
			cudaErrorLaunchFailure => {panic!("
			cudaErrorCooperativeLaunchTooLarge => {panic!("
			cudaErrorNotPermitted => {panic!("
			cudaErrorNotSupported => {panic!("
			cudaErrorSystemNotReady => {panic!("
			cudaErrorSystemDriverMismatch => {panic!("
			cudaErrorCompatNotSupportedOnDevice => {panic!("
			cudaErrorStreamCaptureUnsupported => {panic!("
			cudaErrorStreamCaptureInvalidated => {panic!("
			cudaErrorStreamCaptureMerge => {panic!("
			cudaErrorStreamCaptureUnmatched => {panic!("
			cudaErrorStreamCaptureUnjoined => {panic!("
			cudaErrorStreamCaptureIsolation => {panic!("
			cudaErrorStreamCaptureImplicit => {panic!("
			cudaErrorCapturedEvent => {panic!("
			cudaErrorStreamCaptureWrongThread => {panic!("
			cudaErrorTimeout => {panic!("
			cudaErrorGraphExecUpdateFailure => {panic!("
			cudaErrorUnknown => {panic!("
			cudaErrorApiFailureBase => {panic!("
		}*/		
	}
}

// the pattern/idea of wrapping these types in structures and then catching
// their deletion with Drop is inspired by: https://github.com/autumnai/rust-cudnn/blob/master/cudnn/src/tensor_descriptor.rs (Accessed: Feb 7, 2020)

pub struct Handle {pub cudnn_val: cudnnHandle_t, pub cublas_val: cublasHandle_t}
pub struct cublasHandle {pub val: cublasHandle_t}
pub struct ActivationDescriptor {pub val: cudnnActivationDescriptor_t}
pub struct PoolingDescriptor {pub val: cudnnPoolingDescriptor_t}
pub struct TensorDescriptor {pub val: cudnnTensorDescriptor_t}
pub struct OpTensorDescriptor {pub val: cudnnOpTensorDescriptor_t}
pub struct ReduceTensorDescriptor {pub val: cudnnReduceTensorDescriptor_t}
pub struct FilterDescriptor {pub val: cudnnFilterDescriptor_t}
pub struct ConvolutionDescriptor {pub val: cudnnConvolutionDescriptor_t}
pub struct RNNDescriptor {
	pub val: cudnnRNNDescriptor_t,
	pub dropout_desc: DropoutDescriptor
}
pub struct RNNDataDescriptor {pub val: cudnnRNNDataDescriptor_t}
pub struct DropoutDescriptor {
	pub val: cudnnDropoutDescriptor_t,
	pub dropout_state: gpuMem
}
pub struct SeqDataDescriptor {pub val: cudnnSeqDataDescriptor_t}
pub struct gpuMem {
	pub val: raw::gpuMem_t,
	pub n_elements: size_t,
	pub dataType: cudnnDataType_t,
	pub bytes: size_t
}

impl Drop for Handle {
	fn drop(&mut self) {
		unsafe {raw::cudnnDestroy(self.cudnn_val)}.chk_err();
		unsafe {raw::cublasDestroy_v2(self.cublas_val)}.chk_err();
}}

impl Drop for ActivationDescriptor {
	fn drop(&mut self) {
		unsafe {raw::cudnnDestroyActivationDescriptor(self.val)}.chk_err();
}}

impl Drop for PoolingDescriptor {
	fn drop(&mut self) {
		unsafe {raw::cudnnDestroyPoolingDescriptor(self.val)}.chk_err();
}}

impl Drop for TensorDescriptor {
	fn drop(&mut self) {
		unsafe {raw::cudnnDestroyTensorDescriptor(self.val)}.chk_err();
}}

impl Drop for OpTensorDescriptor {
	fn drop(&mut self) {
		unsafe {raw::cudnnDestroyOpTensorDescriptor(self.val)}.chk_err();
}}

impl Drop for ReduceTensorDescriptor {
	fn drop(&mut self) {
		unsafe {raw::cudnnDestroyReduceTensorDescriptor(self.val)}.chk_err();
}}

impl Drop for FilterDescriptor {
	fn drop(&mut self) {
		unsafe {raw::cudnnDestroyFilterDescriptor(self.val)}.chk_err();
}}

impl Drop for ConvolutionDescriptor {
	fn drop(&mut self) {
		unsafe {raw::cudnnDestroyConvolutionDescriptor(self.val)}.chk_err();
}}

impl Drop for RNNDescriptor {
	fn drop(&mut self) {
		unsafe {raw::cudnnDestroyRNNDescriptor(self.val)}.chk_err();
}}

impl Drop for RNNDataDescriptor {
	fn drop(&mut self) {
		unsafe {raw::cudnnDestroyRNNDataDescriptor(self.val)}.chk_err();
}}

impl Drop for DropoutDescriptor {
	fn drop(&mut self) {
		unsafe {raw::cudnnDestroyDropoutDescriptor(self.val)}.chk_err();
}}

impl Drop for SeqDataDescriptor {
	fn drop(&mut self) {
		unsafe {raw::cudnnDestroySeqDataDescriptor(self.val)}.chk_err();
}}

impl Drop for gpuMem {
	fn drop(&mut self) {
		unsafe {raw::cudaFree(self.val)}.chk_err();
	}
}

impl Handle {
	pub fn new() -> Self {
		let mut cudnn_val = null_mut();
		let mut cublas_val = null_mut();
		
		unsafe {raw::cudnnCreate(&mut cudnn_val)}.chk_err();
		unsafe {raw::cublasCreate_v2(&mut cublas_val)}.chk_err();
		
		Self {cudnn_val, cublas_val}
	}
}

impl ActivationDescriptor {
	fn create() -> Self {
		let mut val = null_mut();
		unsafe {raw::cudnnCreateActivationDescriptor(&mut val)}.chk_err();
		Self {val}
	}
	
	pub fn set(&mut self, mode: cudnnActivationMode_t,
			reluNanOpt: cudnnNanPropagation_t,
			coef: c_double) {
		unsafe {raw::cudnnSetActivationDescriptor(self.val, mode, reluNanOpt, coef)}.chk_err();
	}
	
	pub fn new(mode: cudnnActivationMode_t,
			reluNanOpt: cudnnNanPropagation_t,
			coef: c_double) -> Self {
		let mut desc = Self::create();
		desc.set(mode, reluNanOpt, coef);
		desc
	}
}

impl PoolingDescriptor {
	fn create() -> Self {
		let mut val = null_mut();
		unsafe {raw::cudnnCreatePoolingDescriptor(&mut val)}.chk_err();
		Self {val}
	}
	
	pub fn set(&mut self,
			mode: cudnnPoolingMode_t, maxpoolingNanOpt: cudnnNanPropagation_t,
			windowHeight: c_int, windowWidth: c_int,
			verticalPadding: c_int, horizontalPadding: c_int,
			verticalStride: c_int, horizontalStride: c_int) {
		unsafe {raw::cudnnSetPooling2dDescriptor(self.val, mode, maxpoolingNanOpt,
				windowHeight, windowWidth, verticalPadding,
				horizontalPadding, verticalStride, horizontalStride)}.chk_err();
	}
	
	pub fn new(mode: cudnnPoolingMode_t,
			maxpoolingNanOpt: cudnnNanPropagation_t,
			windowHeight: c_int, windowWidth: c_int,
			verticalPadding: c_int, horizontalPadding: c_int,
			verticalStride: c_int, horizontalStride: c_int) -> Self {
		let mut desc = Self::create();
		
		desc.set(mode, maxpoolingNanOpt, windowHeight, windowWidth, verticalPadding,
				horizontalPadding, verticalStride, horizontalStride);
		desc
	}
}

impl TensorDescriptor {
	fn create() -> Self {
		let mut val = null_mut();
		unsafe {raw::cudnnCreateTensorDescriptor(&mut val)}.chk_err();
		Self {val}
	}
	
	// set strides
	pub fn set_ex(&mut self, dataType: cudnnDataType_t, shape: TensorShape,
			nStride: c_int, cStride: c_int, hStride: c_int, wStride: c_int) {
		unsafe {raw::cudnnSetTensor4dDescriptorEx(self.val, dataType, shape.n, shape.c, 
				shape.h, shape.w, nStride, cStride, hStride, wStride)}.chk_err();
	}
	
	pub fn set(&mut self, format: cudnnTensorFormat_t, dataType: cudnnDataType_t, shape: TensorShape) {
		unsafe {raw::cudnnSetTensor4dDescriptor(self.val, format, dataType, shape.n, shape.c, shape.h, shape.w)}.chk_err();
	}
	
	pub fn set3(&mut self, dataType: cudnnDataType_t, shape: &Tensor3Shape) {
		let dimA = vec![shape.dim1, shape.dim2, shape.dim3, 1];
		let strideA = vec![shape.dim2*shape.dim3, shape.dim3, 1, 1];
		unsafe {raw::cudnnSetTensorNdDescriptor(self.val,
				dataType, 4, dimA.as_ptr(), strideA.as_ptr())
		}.chk_err();
	}
	
	// with strides
	pub fn new_ex(dataType: cudnnDataType_t, shape: TensorShape,
			nStride: c_int, cStride: c_int,
			hStride: c_int, wStride: c_int) -> Self {
		let mut desc = Self::create();
		
		desc.set_ex(dataType, shape, nStride, cStride, hStride, wStride);
		desc
	}
	
	pub fn new(dataType: cudnnDataType_t, format: cudnnTensorFormat_t,
		     shape: TensorShape) -> Self {
		let mut desc = Self::create();
		
		desc.set(format, dataType, shape);
		desc
	}
	
	pub fn new3(dataType: cudnnDataType_t, shape: &Tensor3Shape) -> Self {
		let mut desc = Self::create();
		
		desc.set3(dataType, shape);
		desc
	}
}

impl OpTensorDescriptor {
	fn create() -> Self {
		let mut val = null_mut();
		unsafe {raw::cudnnCreateOpTensorDescriptor(&mut val)}.chk_err();
		Self {val}
	}
	
	pub fn set(&mut self, opTensorOp: cudnnOpTensorOp_t,
			opTensorCompType: cudnnDataType_t,
			opTensorNanOpt: cudnnNanPropagation_t) {
		unsafe {raw::cudnnSetOpTensorDescriptor(self.val, opTensorOp, opTensorCompType, opTensorNanOpt)}.chk_err();
	}
	
	pub fn new(opTensorOp: cudnnOpTensorOp_t,
		     opTensorCompType: cudnnDataType_t,
		     opTensorNanOpt: cudnnNanPropagation_t) -> Self {
		let mut desc = Self::create();
		desc.set(opTensorOp, opTensorCompType, opTensorNanOpt);
		desc
	}
}

impl ReduceTensorDescriptor {
	fn create() -> Self {
		let mut val = null_mut();
		unsafe {raw::cudnnCreateReduceTensorDescriptor(&mut val)}.chk_err();
		Self {val}
	}
	
	pub fn set(&mut self, reduceTensorOp: cudnnReduceTensorOp_t,
		     reduceTensorCompType: cudnnDataType_t,
		     opTensorNanOpt: cudnnNanPropagation_t,
		     reduceTensorIndices: cudnnReduceTensorIndices_t,
		     reduceTensorIndicesType: cudnnIndicesType_t) {
		unsafe {raw::cudnnSetReduceTensorDescriptor(self.val, reduceTensorOp, reduceTensorCompType,
				opTensorNanOpt, reduceTensorIndices, reduceTensorIndicesType)}.chk_err();
	}
	
	pub fn new(reduceTensorOp: cudnnReduceTensorOp_t,
		     reduceTensorCompType: cudnnDataType_t,
		     opTensorNanOpt: cudnnNanPropagation_t,
		     reduceTensorIndices: cudnnReduceTensorIndices_t,
		     reduceTensorIndicesType: cudnnIndicesType_t) -> Self {
		let mut desc = Self::create();
		desc.set(reduceTensorOp, reduceTensorCompType,
				opTensorNanOpt, reduceTensorIndices, reduceTensorIndicesType);
		desc
	}
}

impl FilterDescriptor {
	pub fn create() -> Self {
		let mut val = null_mut();
		unsafe {raw::cudnnCreateFilterDescriptor(&mut val)}.chk_err();
		Self {val}
	}
	
	pub fn set(&mut self, dataType: cudnnDataType_t, format: cudnnTensorFormat_t, shape: FilterShape) {
		unsafe {raw::cudnnSetFilter4dDescriptor(self.val, dataType, format, shape.k, shape.c, shape.h, shape.w)}.chk_err();
	}
	
	pub fn set3(&mut self, dataType: cudnnDataType_t, format: cudnnTensorFormat_t, sz: c_int) {
		let filterDimA = vec![sz, 1, 1];
		unsafe {raw::cudnnSetFilterNdDescriptor(self.val, 
				dataType, format, 3, filterDimA.as_ptr() as *const c_int)}.chk_err();
	}

	pub fn new(dataType: cudnnDataType_t, format: cudnnTensorFormat_t, shape: FilterShape) -> Self {
		let mut desc = Self::create();
		
		desc.set(dataType, format, shape);
		desc
	}
	
	pub fn new3(dataType: cudnnDataType_t, format: cudnnTensorFormat_t, sz: c_int) -> Self {
		let mut desc = Self::create();
		
		desc.set3(dataType, format, sz);
		desc
	}
	
	pub fn shape3(&self) -> Filter3Shape {
		let mut dataType = cudnnDataType_t::CUDNN_DATA_FLOAT;
		let mut format = cudnnTensorFormat_t::CUDNN_TENSOR_NCHW;
		let mut nbDims = 0;
		let mut dims = vec!{0 as c_int; 3};
		
		unsafe {cudnnGetFilterNdDescriptor(self.val, 3,
				&mut dataType, &mut format, &mut nbDims,
				dims.as_mut_ptr())}.chk_err();
		
		debug_assert!(nbDims == 3, "expected 3 filter dims, found: {}", nbDims);
		
		Filter3Shape {dim1: dims[0], dim2: dims[1], dim3: dims[2]}
	}
}

impl ConvolutionDescriptor {
	fn create() -> Self {
		let mut val = null_mut();
		unsafe {raw::cudnnCreateConvolutionDescriptor(&mut val)}.chk_err();
		Self {val}
	}
	
	pub fn set(&mut self, pad_h: c_int, pad_w: c_int, u: c_int, v: c_int,
			dilation_h: c_int, dilation_w: c_int, mode: cudnnConvolutionMode_t,
			computeType: cudnnDataType_t) {
		unsafe {raw::cudnnSetConvolution2dDescriptor(self.val, pad_h, pad_w, u, v, dilation_h, dilation_w,
				mode, computeType)}.chk_err();
	}
	
	pub fn new(pad_h: c_int, pad_w: c_int, u: c_int, v: c_int, dilation_h: c_int, dilation_w: c_int,
			mode: cudnnConvolutionMode_t, computeType: cudnnDataType_t) -> Self {
		let mut desc = Self::create();
		
		desc.set(pad_h, pad_w, u, v, dilation_h, dilation_w, mode, computeType);
		desc
	}
}

impl RNNDescriptor {
	fn create(dropout_desc: DropoutDescriptor) -> Self {
		let mut val = null_mut();
		unsafe {raw::cudnnCreateRNNDescriptor(&mut val)}.chk_err();
		Self {val, dropout_desc}
	}
	
	pub fn set(&mut self, handle: cudnnHandle_t, hiddenSize: c_int, numLayers: c_int,
			inputMode: cudnnRNNInputMode_t,
			direction: cudnnDirectionMode_t, mode: cudnnRNNMode_t,
			algo: cudnnRNNAlgo_t, mathPrec: cudnnDataType_t) {
		unsafe {raw::cudnnSetRNNDescriptor(handle, self.val, hiddenSize, 
				numLayers, self.dropout_desc.val, inputMode, direction, mode, algo,
				mathPrec)}.chk_err();
	}
	
	pub fn new(handle: &Handle, hiddenSize: c_int, numLayers: c_int,
			dropout_desc: DropoutDescriptor, inputMode: cudnnRNNInputMode_t,
			direction: cudnnDirectionMode_t, mode: cudnnRNNMode_t,
			algo: cudnnRNNAlgo_t, mathPrec: cudnnDataType_t) -> Self {
		let mut desc = Self::create(dropout_desc);
		
		desc.set(handle.cudnn_val, hiddenSize, numLayers, inputMode, direction, mode,
				algo, mathPrec);
		desc
	}
}

impl RNNDataDescriptor {
	fn create() -> Self {
		let mut val = null_mut();
		unsafe {raw::cudnnCreateRNNDataDescriptor(&mut val)}.chk_err();
		Self {val}
	}
	
	pub fn set(&mut self, dataType: cudnnDataType_t,
			layout: cudnnRNNDataLayout_t, maxSeqLength: c_int,
			batchSize: c_int, vectorSize: c_int,
			seqLengthArray: &Vec<c_int>
		) {
		debug_assert!(dataType == cudnnDataType_t::CUDNN_DATA_FLOAT); // need to update pad value if this changes
		
		let mut zero_f32 = vec![0. as f32];
		
		let err = unsafe {raw::cudnnSetRNNDataDescriptor(self.val, dataType, layout,
				maxSeqLength, batchSize, vectorSize,
				seqLengthArray.as_ptr() as *const c_int,
				zero_f32.as_mut_ptr() as *mut c_void)};
		
		if err != cudnnStatus_t::CUDNN_STATUS_SUCCESS {
			println!("max_seq_len {} batch_sz {} vec_sz {}", maxSeqLength, batchSize, vectorSize);
			println!("seq_len_array sz {}", seqLengthArray.len());
			for s in seqLengthArray {
				println!("{}", s);
			}
			panic!("failed on cudnnSetRNNDataDescriptor");
		}
		
		/*unsafe {raw::cudnnSetRNNDataDescriptor(self.val, dataType, layout,
				maxSeqLength, batchSize, vectorSize,
				seqLengthArray.as_ptr() as *const c_int,
				zero_f32.as_mut_ptr() as *mut c_void)}.chk_err();*/
	}
	
	pub fn new(dataType: cudnnDataType_t,
			layout: cudnnRNNDataLayout_t, maxSeqLength: c_int,
			batchSize: c_int, vectorSize: c_int,
			seqLengthArray: &Vec<c_int> // c_int
		    ) -> Self {
		let mut desc = Self::create();
		
		desc.set(dataType, layout, maxSeqLength, batchSize,
				vectorSize, seqLengthArray);
		desc
	}
}

impl DropoutDescriptor {
	fn create(handle: &Handle) -> Self {
		// dropout mem
		let dropout_state = {
			let mut dropout_state_sz: size_t = 0;
			unsafe {cudnnDropoutGetStatesSize(handle.cudnn_val, &mut dropout_state_sz)}.chk_err();
			gpuMem::new_bytes(dropout_state_sz)
		};
		
		let mut val = null_mut();
		unsafe {raw::cudnnCreateDropoutDescriptor(&mut val)}.chk_err();
		Self {val, dropout_state}
	}
	
	pub fn set(&mut self, handle: cudnnHandle_t, dropout: c_float, seed: c_ulonglong) {
		unsafe {raw::cudnnSetDropoutDescriptor(self.val, handle, dropout, 
				self.dropout_state.val,
				self.dropout_state.bytes,
				seed)}.chk_err();
	}
	
	pub fn new(handle: &Handle, dropout: c_float, seed: c_ulonglong) -> Self {
		let mut desc = Self::create(handle);
		
		desc.set(handle.cudnn_val, dropout, seed);
		desc
	}
}

impl SeqDataDescriptor {
	/*fn create() -> Self {
		let mut val = null_mut();
		unsafe {raw::cudnnCreateSeqDataDescriptor(&mut val)}.chk_err();
		Self {val}
	}*/
}

// return values for gpu pointer
pub fn ret_raw<T: Default + Clone>(gpu_vals: &gpuMem_t, sz: size_t) -> Vec<T> {
	let mut vals = vec!{Default::default(); sz};
	
	unsafe {cudaMemcpy(vals.as_mut_ptr() as *mut c_void,
			*gpu_vals as *const c_void,
			sz * size_of::<T>(),
			cudaMemcpyKind::cudaMemcpyDeviceToHost
	)}.chk_err();
	
	vals
}

// set values for gpu pointer
pub fn set_raw(gpu_vals: &mut gpuMem_t, vals: &Vec<f32>, sz: size_t) {
	unsafe {cudaMemcpy(*gpu_vals as *mut c_void,
			vals.as_ptr() as *const c_void,
			sz * size_of::<f32>(),
			cudaMemcpyKind::cudaMemcpyHostToDevice
	)}.chk_err();
}

impl gpuMem {
	pub fn new(dataType: cudnnDataType_t, n_elements: size_t) -> Self {
		let mut val = null_mut();
		let bytes = n_elements * dataType.bytes();
		unsafe {raw::cudaMalloc(&mut val, bytes)}.chk_err();
		Self {val, n_elements, dataType, bytes}
	}
	
	pub fn new_bytes(sz: size_t) -> Self {
		let mut val = null_mut();
		const BYTE_TYPE: cudnnDataType_t = cudnnDataType_t::CUDNN_DATA_INT8;
		debug_assert!(BYTE_TYPE.bytes() == 1);
		unsafe {raw::cudaMalloc(&mut val, sz)}.chk_err();
		Self {
			val,
			n_elements: sz,
			dataType: BYTE_TYPE,
			bytes: sz
		}
	}
	
	pub fn set<T>(&self, src: &Vec<T>) {
		debug_assert!(src.len() == self.n_elements);
		debug_assert!(self.dataType.bytes() == size_of::<T>());
		
		unsafe {cudaMemcpy(
				self.val as *mut c_void,
				src.as_ptr() as *const c_void,
				self.bytes,
				cudaMemcpyKind::cudaMemcpyHostToDevice
		)}.chk_err();
	}
	
	pub fn set_underfilled<T>(&self, src: &Vec<T>) {
		debug_assert!(src.len() <= self.n_elements);
		debug_assert!(self.dataType.bytes() == size_of::<T>());
		debug_assert!((src.len() * self.dataType.bytes()) <= self.bytes);
		//println!("{} {}", src.len(), self.n_elements);
		unsafe {cudaMemcpy(
				self.val as *mut c_void,
				src.as_ptr() as *const c_void,
				src.len() * self.dataType.bytes(),
				cudaMemcpyKind::cudaMemcpyHostToDevice
		)}.chk_err();
	}
	
	
	pub fn ret(&self, n_elements: size_t) -> Vec<f32> {
		match self.dataType {
			cudnnDataType_t::CUDNN_DATA_FLOAT => {
				let mut vals = vec!{0.; n_elements};
				self.get(&mut vals);
				vals
			} cudnnDataType_t::CUDNN_DATA_HALF => {
				let mut vals = vec![0 as f16; n_elements];
				self.get(&mut vals);
				
				let mut vals_conv = Vec::with_capacity(n_elements);
				for v in vals {
					vals_conv.push(f16_to_f32(v));
				}
				vals_conv
			} _ => {panic!("unsupported data type");}
		}
	}
	
	pub fn get<T>(&self, dest: &mut Vec<T>) {
		debug_assert!(dest.len() <= self.n_elements);
		debug_assert!(self.dataType.bytes() == size_of::<T>());
		
		unsafe {cudaMemcpy(
				dest.as_mut_ptr() as *mut c_void,
				self.val as *const c_void,
				dest.len() * size_of::<T>(),
				cudaMemcpyKind::cudaMemcpyDeviceToHost
		)}.chk_err();
	}
	
	pub fn init<T>(dataType: cudnnDataType_t, vals: &Vec<T>) -> Self {
		let gpu_mem = gpuMem::new(dataType, vals.len());
		gpu_mem.set(vals);
		gpu_mem
	}
	
	pub fn zero_out(&self) {
		unsafe {cudaMemset(self.val, 0, self.bytes)}.chk_err();
	}
	
	pub fn one_out(&self) {
		self.set(&vec!{1. as f32; self.n_elements});
	}
}

pub fn set_device(gpu_ind: c_int) {
	unsafe {raw::cudaSetDevice(gpu_ind)}.chk_err();
}

// get which gpu to run on from cmd line, otherwise, run on device 0
pub fn set_device_from_cmd_args() {
	let args = env::args().collect::<Vec<String>>();
	
	// card should be specified on the command line
	let gpu_card = if args.len() > 1 {
		let gpu_card_arg = &args[1];
		if let Result::Ok(gpu_card) = gpu_card_arg.parse() {
			gpu_card
		}else{panic!("Cannot parse command line input: \"{}\"", gpu_card_arg)}
	// card was not specified on the command line
	}else {0};
	
	println!("running on {}", gpu_card);
	set_device(gpu_card);
}

impl cudnnDataType_t {
	pub fn bytes(&self) -> size_t {
		match self {
			cudnnDataType_t::CUDNN_DATA_FLOAT => 4,
			cudnnDataType_t::CUDNN_DATA_DOUBLE => 8,
			cudnnDataType_t::CUDNN_DATA_HALF => 2,
			cudnnDataType_t::CUDNN_DATA_INT8 => 1,
			cudnnDataType_t::CUDNN_DATA_INT32 => 4,
			cudnnDataType_t::CUDNN_DATA_INT8x4 => 4,
			cudnnDataType_t::CUDNN_DATA_UINT8 => 1,
			cudnnDataType_t::CUDNN_DATA_UINT8x4 => 4,
			cudnnDataType_t::CUDNN_DATA_INT8x32 => 32
		}
	}
}

pub fn used_bytes() -> size_t {
	let mut free: size_t = 0; //vec![0 as size_t];
	let mut total: size_t = 0; //vec![0 as size_t];
	
	unsafe {cudaMemGetInfo(&mut free, &mut total)}.chk_err();
	
	total - free
}

