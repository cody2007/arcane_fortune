// See: (Feb 7, 2020)
//	/usr/include/cudnn.h
//	/usr/local/cuda/include/cuda_runtime.h
//		/usr/local/cuda/include/cuda_runtime_api.h
//		/usr/local/cuda/include/driver_types.h
//
// Cudnn version: 7.6.4; Feb 7, 2020
// Cuda nvidia-smi output: Feb 7, 2020
// 		| NVIDIA-SMI 440.33.01 Driver Version: 440.33.01 CUDA Version: 10.2 |


// test hardware:
// (0): GeForce RTX 2080 Ti, Compute Capability 7.5
// (1): GeForce GTX TITAN X, Compute Capability 5.2

#![allow(non_camel_case_types)]

use std::ffi::c_void;
use std::os::raw::{c_int, c_float, c_longlong, c_ulonglong, c_double};
use crate::data_wrappers::*;
//use super::f16;
//use crate::cudnn_common::*; // also should be updated if cudnn.h definitions change

// see /usr/local/cuda/include/driver_types.h (Feb 7, 2020)
#[derive(PartialEq, Clone, Copy)]
#[repr(C)]
pub enum cudaError_t {
	cudaSuccess = 0,
	cudaErrorInvalidValue = 1,
	cudaErrorMemoryAllocation = 2,
	cudaErrorInitializationError = 3,
	cudaErrorCudartUnloading = 4,
	cudaErrorProfilerDisabled = 5,
	cudaErrorProfilerNotInitialized = 6,
	cudaErrorProfilerAlreadyStarted = 7,
	cudaErrorProfilerAlreadyStopped = 8,
	cudaErrorInvalidConfiguration = 9,
	cudaErrorInvalidPitchValue = 12,
	cudaErrorInvalidSymbol = 13,
	cudaErrorInvalidHostPointer = 16,
	cudaErrorInvalidDevicePointer = 17,
	cudaErrorInvalidTexture = 18,
	cudaErrorInvalidTextureBinding = 19,
	cudaErrorInvalidChannelDescriptor = 20,
	cudaErrorInvalidMemcpyDirection = 21,
	cudaErrorAddressOfConstant = 22,
	cudaErrorTextureFetchFailed = 23,
	cudaErrorTextureNotBound = 24,
	cudaErrorSynchronizationError = 25,
	cudaErrorInvalidFilterSetting = 26,
	cudaErrorInvalidNormSetting = 27,
	cudaErrorMixedDeviceExecution = 28,
	cudaErrorNotYetImplemented = 31,
	cudaErrorMemoryValueTooLarge = 32,
	cudaErrorInsufficientDriver = 35,
	cudaErrorInvalidSurface = 37,
	cudaErrorDuplicateVariableName = 43,
	cudaErrorDuplicateTextureName = 44,
	cudaErrorDuplicateSurfaceName = 45,
	cudaErrorDevicesUnavailable = 46,
	cudaErrorIncompatibleDriverContext = 49,
	cudaErrorMissingConfiguration = 52,
	cudaErrorPriorLaunchFailure = 53,
	cudaErrorLaunchMaxDepthExceeded = 65,
	cudaErrorLaunchFileScopedTex = 66,
	cudaErrorLaunchFileScopedSurf = 67,
	cudaErrorSyncDepthExceeded = 68,
	cudaErrorLaunchPendingCountExceeded = 69,
	cudaErrorInvalidDeviceFunction = 98,
	cudaErrorNoDevice = 100,
	cudaErrorInvalidDevice = 101,
	cudaErrorStartupFailure = 127,
	cudaErrorInvalidKernelImage = 200,
	cudaErrorDeviceUninitialized = 201,
	cudaErrorMapBufferObjectFailed = 205,
	cudaErrorUnmapBufferObjectFailed = 206,
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
	cudaErrorPeerAccessAlreadyEnabled = 704,
	cudaErrorPeerAccessNotEnabled = 705,
	cudaErrorSetOnActiveProcess = 708,
	cudaErrorContextIsDestroyed = 709,
	cudaErrorAssert = 710,
	cudaErrorTooManyPeers = 711,
	cudaErrorHostMemoryAlreadyRegistered = 712,
	cudaErrorHostMemoryNotRegistered = 713,
	cudaErrorHardwareStackError = 714,
	cudaErrorIllegalInstruction = 715,
	cudaErrorMisalignedAddress = 716,
	cudaErrorInvalidAddressSpace = 717,
	cudaErrorInvalidPc = 718,
	cudaErrorLaunchFailure = 719,
	cudaErrorCooperativeLaunchTooLarge = 720,
	cudaErrorNotPermitted = 800,
	cudaErrorNotSupported = 801,
	cudaErrorSystemNotReady = 802,
	cudaErrorSystemDriverMismatch = 803,
	cudaErrorCompatNotSupportedOnDevice = 804,
	cudaErrorStreamCaptureUnsupported = 900,
	cudaErrorStreamCaptureInvalidated = 901,
	cudaErrorStreamCaptureMerge = 902,
	cudaErrorStreamCaptureUnmatched = 903,
	cudaErrorStreamCaptureUnjoined = 904,
	cudaErrorStreamCaptureIsolation = 905,
	cudaErrorStreamCaptureImplicit = 906,
	cudaErrorCapturedEvent = 907,
	cudaErrorStreamCaptureWrongThread = 908,
	cudaErrorTimeout = 909,
	cudaErrorGraphExecUpdateFailure = 910,
	cudaErrorUnknown = 999,
	cudaErrorApiFailureBase = 10000
}

#[derive(PartialEq)]
#[repr(C)]
pub enum cudnnStatus_t {
	CUDNN_STATUS_SUCCESS = 0,
	CUDNN_STATUS_NOT_INITIALIZED = 1,
	CUDNN_STATUS_ALLOC_FAILED = 2,
	CUDNN_STATUS_BAD_PARAM = 3,
	CUDNN_STATUS_INTERNAL_ERROR = 4,
	CUDNN_STATUS_INVALID_VALUE = 5,
	CUDNN_STATUS_ARCH_MISMATCH = 6,
	CUDNN_STATUS_MAPPING_ERROR = 7,
	CUDNN_STATUS_EXECUTION_FAILED = 8,
	CUDNN_STATUS_NOT_SUPPORTED = 9,
	CUDNN_STATUS_LICENSE_ERROR = 10,
	CUDNN_STATUS_RUNTIME_PREREQUISITE_MISSING = 11,
	CUDNN_STATUS_RUNTIME_IN_PROGRESS = 12,
	CUDNN_STATUS_RUNTIME_FP_OVERFLOW = 13
}

#[repr(C)]
pub enum cudnnPoolingMode_t {
	CUDNN_POOLING_MAX = 0,
	CUDNN_POOLING_AVERAGE_COUNT_INCLUDE_PADDING = 1,
	CUDNN_POOLING_AVERAGE_COUNT_EXCLUDE_PADDING = 2,
	CUDNN_POOLING_MAX_DETERMINISTIC = 3
}

#[repr(C)]
pub enum cudnnNanPropagation_t {
	CUDNN_NOT_PROPAGATE_NAN = 0,
	CUDNN_PROPAGATE_NAN = 1
}

#[derive(Copy, Clone)]
#[repr(C)]
pub enum cudnnActivationMode_t {
	CUDNN_ACTIVATION_SIGMOID = 0,
	CUDNN_ACTIVATION_RELU = 1,
	CUDNN_ACTIVATION_TANH = 2,
	CUDNN_ACTIVATION_CLIPPED_RELU = 3,
	CUDNN_ACTIVATION_ELU = 4,
	CUDNN_ACTIVATION_IDENTITY = 5
}

#[repr(C)]
pub enum cudnnConvolutionMode_t {
	CUDNN_CONVOLUTION = 0,
	CUDNN_CROSS_CORRELATION = 1
}

#[derive(Copy, Clone)]
#[repr(C)]
pub enum cudnnConvolutionFwdAlgo_t {
	CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM = 0,
	CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM = 1,
	CUDNN_CONVOLUTION_FWD_ALGO_GEMM = 2,
	CUDNN_CONVOLUTION_FWD_ALGO_DIRECT = 3,
	CUDNN_CONVOLUTION_FWD_ALGO_FFT = 4,
	CUDNN_CONVOLUTION_FWD_ALGO_FFT_TILING = 5,
	CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD = 6,
	CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD_NONFUSED = 7,
	CUDNN_CONVOLUTION_FWD_ALGO_COUNT = 8
}

#[derive(Copy, Clone)]
#[repr(C)]
pub enum cudnnConvolutionBwdDataAlgo_t {
	CUDNN_CONVOLUTION_BWD_DATA_ALGO_0 = 0,
	CUDNN_CONVOLUTION_BWD_DATA_ALGO_1 = 1,
	CUDNN_CONVOLUTION_BWD_DATA_ALGO_FFT = 2,
	CUDNN_CONVOLUTION_BWD_DATA_ALGO_FFT_TILING = 3,
	CUDNN_CONVOLUTION_BWD_DATA_ALGO_WINOGRAD = 4,
	CUDNN_CONVOLUTION_BWD_DATA_ALGO_WINOGRAD_NONFUSED = 5,
	CUDNN_CONVOLUTION_BWD_DATA_ALGO_COUNT = 6
}

#[derive(Copy, Clone)]
#[repr(C)]
pub enum cudnnConvolutionBwdFilterAlgo_t {
	CUDNN_CONVOLUTION_BWD_FILTER_ALGO_0 = 0,
	CUDNN_CONVOLUTION_BWD_FILTER_ALGO_1 = 1,
	CUDNN_CONVOLUTION_BWD_FILTER_ALGO_FFT = 2,
	CUDNN_CONVOLUTION_BWD_FILTER_ALGO_3 = 3,
	CUDNN_CONVOLUTION_BWD_FILTER_ALGO_WINOGRAD = 4,
	CUDNN_CONVOLUTION_BWD_FILTER_ALGO_WINOGRAD_NONFUSED = 5,
	CUDNN_CONVOLUTION_BWD_FILTER_ALGO_FFT_TILING = 6,
	CUDNN_CONVOLUTION_BWD_FILTER_ALGO_COUNT = 7
}

#[repr(C)]
pub enum cudnnConvolutionFwdPreference_t {
	CUDNN_CONVOLUTION_FWD_NO_WORKSPACE = 0,
	CUDNN_CONVOLUTION_FWD_PREFER_FASTEST = 1,
	CUDNN_CONVOLUTION_FWD_SPECIFY_WORKSPACE_LIMIT = 2
}

#[repr(C)]
pub enum cudnnConvolutionBwdDataPreference_t {
	CUDNN_CONVOLUTION_BWD_DATA_NO_WORKSPACE = 0,
	CUDNN_CONVOLUTION_BWD_DATA_PREFER_FASTEST = 1,
	CUDNN_CONVOLUTION_BWD_DATA_SPECIFY_WORKSPACE_LIMIT = 2
}

#[repr(C)]
pub enum cudnnConvolutionBwdFilterPreference_t {
	CUDNN_CONVOLUTION_BWD_FILTER_NO_WORKSPACE = 0,
	CUDNN_CONVOLUTION_BWD_FILTER_PREFER_FASTEST = 1,
	CUDNN_CONVOLUTION_BWD_FILTER_SPECIFY_WORKSPACE_LIMIT = 2
}

#[derive(Copy, Clone)]
#[repr(C)]
pub enum cudnnRNNAlgo_t {
	CUDNN_RNN_ALGO_STANDARD = 0,
	CUDNN_RNN_ALGO_PERSIST_STATIC = 1,
	CUDNN_RNN_ALGO_PERSIST_DYNAMIC = 2,
	CUDNN_RNN_ALGO_COUNT = 3,
}

#[repr(C)]
pub enum cudnnSoftmaxMode_t {
	CUDNN_SOFTMAX_MODE_INSTANCE = 0, // documentation: "compute the softmax over all C, H, W for each N" (cudnn.h)
	CUDNN_SOFTMAX_MODE_CHANNEL = 1 // documentation: "compute the softmax over all C for each H, W, N" (cudnn.h)
}

#[repr(C)]
pub enum cudnnRNNMode_t {
	CUDNN_RNN_RELU = 0,
	CUDNN_RNN_TANH = 1,
	CUDNN_LSTM = 2,
	CUDNN_GRU = 3
}

#[repr(C)]
pub enum cudnnRNNBiasMode_t {
	CUDNN_RNN_NO_BIAS = 0, // "rnn cell formulas do not use biases" (cudnn.h)
	CUDNN_RNN_SINGLE_INP_BIAS = 1, // "rnn cell formulas use one input bias in input GEMM" (cudnn.h)
	CUDNN_RNN_DOUBLE_BIAS = 2, // "default, rnn cell formulas use two bias vectors" (cudnn.h)
	CUDNN_RNN_SINGLE_REC_BIAS = 3  // "rnn cell formulas use one recurrent bias in recurrent GEMM" (cudnn.h)
}

#[repr(C)]
pub enum cudnnRNNClipMode_t {
	CUDNN_RNN_CLIP_NONE = 0,
	CUDNN_RNN_CLIP_MINMAX = 1 
}

#[repr(C)]
pub enum cudnnRNNPaddingMode_t {
	CUDNN_RNN_PADDED_IO_DISABLED = 0,
	CUDNN_RNN_PADDED_IO_ENABLED = 1
}

#[repr(C)]
pub enum cudnnRNNInputMode_t {
	CUDNN_LINEAR_INPUT = 0,
	CUDNN_SKIP_INPUT = 1
}

#[repr(C)]
pub enum cudnnOpTensorOp_t {
	CUDNN_OP_TENSOR_ADD = 0,
	CUDNN_OP_TENSOR_MUL = 1,
	CUDNN_OP_TENSOR_MIN = 2,
	CUDNN_OP_TENSOR_MAX = 3,
	CUDNN_OP_TENSOR_SQRT = 4,
	CUDNN_OP_TENSOR_NOT = 5
}

#[repr(C)]
pub enum cudnnReduceTensorOp_t {
	CUDNN_REDUCE_TENSOR_ADD = 0,
	CUDNN_REDUCE_TENSOR_MUL = 1,
	CUDNN_REDUCE_TENSOR_MIN = 2,
	CUDNN_REDUCE_TENSOR_MAX = 3,
	CUDNN_REDUCE_TENSOR_AMAX = 4,
	CUDNN_REDUCE_TENSOR_AVG = 5,
	CUDNN_REDUCE_TENSOR_NORM1 = 6,
	CUDNN_REDUCE_TENSOR_NORM2 = 7,
	CUDNN_REDUCE_TENSOR_MUL_NO_ZEROS = 8,
}

#[repr(C)]
pub enum cudnnReduceTensorIndices_t {
	CUDNN_REDUCE_TENSOR_NO_INDICES = 0,
	CUDNN_REDUCE_TENSOR_FLATTENED_INDICES = 1,
}

#[repr(C)]
pub enum cudnnIndicesType_t { // all unsigned, only 32bit unsigned is supported
	CUDNN_32BIT_INDICES = 0,
	CUDNN_64BIT_INDICES = 1,
	CUDNN_16BIT_INDICES = 2,
	CUDNN_8BIT_INDICES = 3
}

#[repr(C)]
pub enum cudnnDirectionMode_t {
	CUDNN_UNIDIRECTIONAL = 0, // "The network iterates recurrently from the first input to the last." - pg. 14, cuDNN API DA-09702-001_v7.6.5
	CUDNN_BIDIRECTIONAL = 1
}

#[repr(C)]
pub enum cudnnSeqDataAxis_t {
	CUDNN_SEQDATA_TIME_DIM = 0,
	CUDNN_SEQDATA_BATCH_DIM = 1,
	CUDNN_SEQDATA_BEAM_DIM = 2,
	CUDNN_SEQDATA_VECT_DIM = 3
}

pub const CUDNN_SEQDATA_DIM_COUNT: usize = 4;

// idea to use raw pointers to null enums: https://github.com/autumnai/rust-cudnn/blob/master/cudnn-sys/src/lib.rs, Accessed March 7, 2020
pub enum cudnnContext {}
pub type cudnnHandle_t = *mut cudnnContext;

pub enum cublasContext {}
pub type cublasHandle_t = *mut cublasContext;

pub enum cudnnActivationStruct {}
pub enum cudnnPoolingStruct {}
pub enum cudnnTensorStruct {}
pub enum cudnnOpTensorStruct {}
pub enum cudnnReduceTensorStruct {}
pub enum cudnnFilterStruct {}
pub enum cudnnConvolutionStruct {}
pub enum cudnnRNNStruct {}
pub enum cudnnRNNDataStruct {}
pub enum cudnnSeqDataStruct {}
pub enum cudnnDropoutStruct {}

pub type cudnnActivationDescriptor_t = *mut cudnnActivationStruct;
pub type cudnnPoolingDescriptor_t = *mut cudnnPoolingStruct;
pub type cudnnTensorDescriptor_t = *mut cudnnTensorStruct;
pub type cudnnOpTensorDescriptor_t = *mut cudnnOpTensorStruct;
pub type cudnnReduceTensorDescriptor_t = *mut cudnnReduceTensorStruct;
pub type cudnnFilterDescriptor_t = *mut cudnnFilterStruct;
pub type cudnnConvolutionDescriptor_t = *mut cudnnConvolutionStruct;
pub type cudnnRNNDescriptor_t = *mut cudnnRNNStruct;
pub type cudnnRNNDataDescriptor_t = *mut cudnnRNNDataStruct;
pub type cudnnSeqDataDescriptor_t = *mut cudnnSeqDataStruct;
pub type cudnnDropoutDescriptor_t = *mut cudnnDropoutStruct;

pub enum gpuMem_Struct {}
pub type gpuMem_t = *mut gpuMem_Struct;

////////////////////////////////////////////////////////////////////////////////////////////////////
// cudnn functions

#[link(name = "cudnn")]
extern "C" {
	////////////////////////
	// create/destroy
	pub fn cudnnCreate(handle: *mut cudnnHandle_t) -> cudnnStatus_t;
	pub fn cudnnDestroy(handle: cudnnHandle_t) -> cudnnStatus_t;
	
	pub fn cudnnCreateActivationDescriptor(activationDesc: *mut cudnnActivationDescriptor_t) -> cudnnStatus_t;
	pub fn cudnnDestroyActivationDescriptor(activationDesc: cudnnActivationDescriptor_t) -> cudnnStatus_t;
	
	pub fn cudnnCreatePoolingDescriptor(poolingDesc: *mut cudnnPoolingDescriptor_t) -> cudnnStatus_t;
	pub fn cudnnDestroyPoolingDescriptor(poolingDesc: cudnnPoolingDescriptor_t) -> cudnnStatus_t;
	
	pub fn cudnnCreateTensorDescriptor(tensorDesc: *mut cudnnTensorDescriptor_t) -> cudnnStatus_t;
	pub fn cudnnDestroyTensorDescriptor(tensorDesc: cudnnTensorDescriptor_t) -> cudnnStatus_t;
	
	pub fn cudnnCreateOpTensorDescriptor(opTensorDesc: *mut cudnnOpTensorDescriptor_t) -> cudnnStatus_t;
	pub fn cudnnDestroyOpTensorDescriptor(opTensorDesc: cudnnOpTensorDescriptor_t) -> cudnnStatus_t;
	
	pub fn cudnnCreateReduceTensorDescriptor(reduceTensorDesc: *mut cudnnReduceTensorDescriptor_t) -> cudnnStatus_t;
	pub fn cudnnDestroyReduceTensorDescriptor(reduceTensorDesc: cudnnReduceTensorDescriptor_t) -> cudnnStatus_t;
	
	pub fn cudnnCreateFilterDescriptor(filterDesc: *mut cudnnFilterDescriptor_t) -> cudnnStatus_t;
	pub fn cudnnDestroyFilterDescriptor(filterDesc: cudnnFilterDescriptor_t) -> cudnnStatus_t;
	
	pub fn cudnnCreateConvolutionDescriptor(convDesc: *mut cudnnConvolutionDescriptor_t) -> cudnnStatus_t;
	pub fn cudnnDestroyConvolutionDescriptor(convDesc: cudnnConvolutionDescriptor_t) -> cudnnStatus_t;
	
	pub fn cudnnCreateRNNDescriptor(rnnDesc: *mut cudnnRNNDescriptor_t) -> cudnnStatus_t;
	pub fn cudnnDestroyRNNDescriptor(rnnDesc: cudnnRNNDescriptor_t) -> cudnnStatus_t;
	
	pub fn cudnnCreateSeqDataDescriptor(seqDataDesc: *mut cudnnSeqDataDescriptor_t) -> cudnnStatus_t;
	pub fn cudnnDestroySeqDataDescriptor(seqDataDesc: cudnnSeqDataDescriptor_t) -> cudnnStatus_t;
	
	pub fn cudnnCreateDropoutDescriptor(dropoutDesc: *mut cudnnDropoutDescriptor_t) -> cudnnStatus_t;
	pub fn cudnnDestroyDropoutDescriptor(dropoutDesc: cudnnDropoutDescriptor_t) -> cudnnStatus_t;
	
	pub fn cudnnCreateRNNDataDescriptor(RNNDataDesc: *mut cudnnRNNDataDescriptor_t) -> cudnnStatus_t;
	pub fn cudnnDestroyRNNDataDescriptor(RNNDataDesc: cudnnRNNDataDescriptor_t) -> cudnnStatus_t;
		
	//////////////////
	// set
	pub fn cudnnSetPooling2dDescriptor(poolingDesc: cudnnPoolingDescriptor_t,
			mode: cudnnPoolingMode_t,
			maxpoolingNanOpt: cudnnNanPropagation_t,
			windowHeight: c_int,
			windowWidth: c_int,
			verticalPadding: c_int,
			horizontalPadding: c_int,
			verticalStride: c_int,
			horizontalStride: c_int) -> cudnnStatus_t;
	
	pub fn cudnnSetActivationDescriptor(activationDesc: cudnnActivationDescriptor_t,
			mode: cudnnActivationMode_t,
			reluNanOpt: cudnnNanPropagation_t,
			coef: c_double) -> cudnnStatus_t;
	
	pub fn cudnnSetTensor4dDescriptor(tensorDesc: cudnnTensorDescriptor_t,
			format: cudnnTensorFormat_t,
			dataType: cudnnDataType_t,
			n: c_int,
			c: c_int,
			h: c_int,
			w: c_int) -> cudnnStatus_t;
	
	pub fn cudnnSetTensor4dDescriptorEx(tensorDesc: cudnnTensorDescriptor_t,
			dataType: cudnnDataType_t,
			n: c_int,
			c: c_int,
			h: c_int,
			w: c_int,
			nStride: c_int,
			cStride: c_int,
			hStride: c_int,
			wStride: c_int) -> cudnnStatus_t;
	
	pub fn cudnnSetTensorNdDescriptor(tensorDesc: cudnnTensorDescriptor_t,
			dataType: cudnnDataType_t,
			nbDims: c_int,
			dimA: *const c_int,
			strideA: *const c_int) -> cudnnStatus_t;
	
	pub fn cudnnSetOpTensorDescriptor(
			opTensorDesc: cudnnOpTensorDescriptor_t,
			opTensorOp: cudnnOpTensorOp_t,
			opTensorCompType: cudnnDataType_t,
			opTensorNanOpt: cudnnNanPropagation_t) -> cudnnStatus_t;
	
	pub fn cudnnSetReduceTensorDescriptor(reduceTensorDesc: cudnnReduceTensorDescriptor_t,
			reduceTensorOp: cudnnReduceTensorOp_t,
			reduceTensorCompType: cudnnDataType_t,
			reduceTensorNanOpt: cudnnNanPropagation_t,
			reduceTensorIndices: cudnnReduceTensorIndices_t,
			reduceTensorIndicesType: cudnnIndicesType_t) -> cudnnStatus_t;
	
	pub fn cudnnSetFilter4dDescriptor(filterDesc: cudnnFilterDescriptor_t,
			dataType: cudnnDataType_t,
			format: cudnnTensorFormat_t,
			k: c_int,
			c: c_int,
			h: c_int,
			w: c_int) -> cudnnStatus_t;
	
	pub fn cudnnSetFilterNdDescriptor(filterDesc: cudnnFilterDescriptor_t,
			dataType: cudnnDataType_t,
			format: cudnnTensorFormat_t,
			nbDims: c_int,
			filterDimA: *const c_int) -> cudnnStatus_t;
	
	pub fn cudnnSetConvolution2dDescriptor(convDesc: cudnnConvolutionDescriptor_t,
			pad_h: c_int,
			pad_w: c_int,
			u: c_int,
			v: c_int,
			dilation_h: c_int,
			dilation_w: c_int,
			mode: cudnnConvolutionMode_t,
			computeType: cudnnDataType_t) -> cudnnStatus_t;
	
	pub fn cudnnSetRNNDescriptor(handle: cudnnHandle_t,
			rnnDesc: cudnnRNNDescriptor_t,
			hiddenSize: c_int,
			numLayers: c_int,
			dropoutDesc: cudnnDropoutDescriptor_t,
			inputMode: cudnnRNNInputMode_t,
			direction: cudnnDirectionMode_t,
			mode: cudnnRNNMode_t,
			algo: cudnnRNNAlgo_t,
			mathPrec: cudnnDataType_t) -> cudnnStatus_t;
	
	pub fn cudnnSetDropoutDescriptor(dropoutDesc: cudnnDropoutDescriptor_t,
			handle: cudnnHandle_t,
			dropout: c_float,
			states: gpuMem_t,
			stateSizeInBytes: size_t,
			seed: c_ulonglong) -> cudnnStatus_t;
	
	pub fn cudnnSetSeqDataDescriptor(seqDataDesc: cudnnSeqDataDescriptor_t,
			dataType: cudnnDataType_t,
			nbDims: c_int,
			dimA: *const c_int,
			axes: *const cudnnSeqDataAxis_t,
			seqLengthArraySize: size_t,
			seqLengthArray: *const c_int,
			paddingFill: *mut c_void) -> cudnnStatus_t;
	
	pub fn cudnnSetRNNDataDescriptor(RNNDataDesc: cudnnRNNDataDescriptor_t,
			dataType: cudnnDataType_t,
			layout: cudnnRNNDataLayout_t,
			maxSeqLength: c_int,
			batchSize: c_int,
			vectorSize: c_int,
			seqLengthArray: *const c_int, // c_int
			paddingFill: *mut c_void 
			// ^ "The symbol should be in the host memory, and is interpreted as
			//    the same data type as that of the RNN data tensor."
			//
			//	"If a NULL pointer is passed in, then
			//	the padding position in the output will be undefined."
			//
			// 		--cuDNN API DA-09702-001_v7.6.5 | pg 330
			) -> cudnnStatus_t;
	
	pub fn cudnnSetRNNPaddingMode(RNNDataDesc: cudnnRNNDescriptor_t,
			paddingMode: cudnnRNNPaddingMode_t) -> cudnnStatus_t;
	
	//////////////// get
	pub fn cudnnGetConvolutionForwardAlgorithm(handle: cudnnHandle_t,
			xDesc: cudnnTensorDescriptor_t,
			wDesc: cudnnFilterDescriptor_t,
			convDesc: cudnnConvolutionDescriptor_t,
			yDesc: cudnnTensorDescriptor_t,
			preference: cudnnConvolutionFwdPreference_t,
			memoryLimitInBytes: size_t,
			algo: *mut cudnnConvolutionFwdAlgo_t) -> cudnnStatus_t;
	
	pub fn cudnnGetConvolutionBackwardDataAlgorithm(handle: cudnnHandle_t,
			wDesc: cudnnFilterDescriptor_t,
			dyDesc: cudnnTensorDescriptor_t,
			convDesc: cudnnConvolutionDescriptor_t,
			dxDesc: cudnnTensorDescriptor_t,
			preference: cudnnConvolutionBwdDataPreference_t,
			memoryLimitInBytes: size_t,
			algo: *mut cudnnConvolutionBwdDataAlgo_t) -> cudnnStatus_t;
	
	pub fn cudnnGetConvolutionBackwardFilterAlgorithm(handle: cudnnHandle_t,
			xDesc: cudnnTensorDescriptor_t,
			dyDesc: cudnnTensorDescriptor_t,
			convDesc: cudnnConvolutionDescriptor_t,
			dwDesc: cudnnFilterDescriptor_t,
			preference: cudnnConvolutionBwdFilterPreference_t,
			memoryLimitInBytes: size_t,
			algo: *mut cudnnConvolutionBwdFilterAlgo_t) -> cudnnStatus_t;
	
	pub fn cudnnGetConvolutionForwardWorkspaceSize(handle: cudnnHandle_t,
			xDesc: cudnnTensorDescriptor_t,
			wDesc: cudnnFilterDescriptor_t,
			convDesc: cudnnConvolutionDescriptor_t,
			yDesc: cudnnTensorDescriptor_t,
			algo: cudnnConvolutionFwdAlgo_t,
			sizeInBytes: *mut size_t) -> cudnnStatus_t;
	
	pub fn cudnnGetConvolutionBackwardDataWorkspaceSize(handle: cudnnHandle_t,
			wDesc: cudnnFilterDescriptor_t,
			dyDesc: cudnnTensorDescriptor_t,
			convDesc: cudnnConvolutionDescriptor_t,
			dxDesc: cudnnTensorDescriptor_t,
			algo: cudnnConvolutionBwdDataAlgo_t,
			sizeInBytes: *mut size_t) -> cudnnStatus_t;
			
	pub fn cudnnGetConvolutionBackwardFilterWorkspaceSize(handle: cudnnHandle_t,
			xDesc: cudnnTensorDescriptor_t,
			dyDesc: cudnnTensorDescriptor_t,
			convDesc: cudnnConvolutionDescriptor_t,
			dwDesc: cudnnFilterDescriptor_t,
			algo: cudnnConvolutionBwdFilterAlgo_t,
			sizeInBytes: *mut size_t) -> cudnnStatus_t;
	
	pub fn cudnnGetReductionWorkspaceSize(handle: cudnnHandle_t,
			reduceDesc: cudnnReduceTensorDescriptor_t,
			aDesc: cudnnTensorDescriptor_t,
			cDesc: cudnnTensorDescriptor_t,
			sizeInBytes: *mut size_t) -> cudnnStatus_t;
	
	pub fn cudnnGetReductionIndicesSize(handle: cudnnHandle_t,
			reduceDesc: cudnnReduceTensorDescriptor_t,
			aDesc: cudnnTensorDescriptor_t,
			cDesc: cudnnTensorDescriptor_t,
			sizeInBytes: *mut size_t) -> cudnnStatus_t;
	
	pub fn cudnnGetRNNWorkspaceSize(handle: cudnnHandle_t,
			rnnDesc: cudnnRNNDescriptor_t,
			seqLength: c_int,
			xDesc: *const cudnnTensorDescriptor_t,
			sizeInBytes: *mut size_t) -> cudnnStatus_t;
	
	pub fn cudnnGetRNNTrainingReserveSize(handle: cudnnHandle_t,
			rnnDesc: cudnnRNNDescriptor_t,
			seqLength: c_int,
			xDesc: *const cudnnTensorDescriptor_t,
			sizeInBytes: *mut size_t) -> cudnnStatus_t;
	
	pub fn cudnnDropoutGetStatesSize(handle: cudnnHandle_t, sizeInBytes: *mut size_t) -> cudnnStatus_t;
	
	pub fn cudnnGetRNNParamsSize(handle: cudnnHandle_t,
			rnnDesc: cudnnRNNDescriptor_t,
			xDesc: cudnnTensorDescriptor_t,
			sizeInBytes: *mut size_t,
			dataType: cudnnDataType_t) -> cudnnStatus_t;
	
	pub fn cudnnGetRNNLinLayerMatrixParams(handle: cudnnHandle_t,
			rnnDesc: cudnnRNNDescriptor_t,
			pseudoLayer: c_int,
			xDesc: cudnnTensorDescriptor_t,
			wDesc: cudnnFilterDescriptor_t,
			w: gpuMem_t,
			linLayerID: c_int,
			linLayerMatDesc: cudnnFilterDescriptor_t,
			linLayerMat: *mut gpuMem_t) -> cudnnStatus_t;
	
	pub fn cudnnGetRNNLinLayerBiasParams(handle: cudnnHandle_t,
			rnnDesc: cudnnRNNDescriptor_t,
			pseudoLayer: c_int,
			xDesc: cudnnTensorDescriptor_t,
			wDesc: cudnnFilterDescriptor_t,
			w: gpuMem_t,
			linLayerID: c_int,
			linLayerBiasDesc: cudnnFilterDescriptor_t,
			linLayerBias: *mut gpuMem_t) -> cudnnStatus_t;
	
	pub fn cudnnGetFilterNdDescriptor(wDesc: cudnnFilterDescriptor_t,
			nbDimsRequested: c_int,
			dataType: &mut cudnnDataType_t,
			format: &mut cudnnTensorFormat_t,
			nbDims: &mut c_int,
			filterDimA: *mut c_int) -> cudnnStatus_t;
	
	////////////////// forward/backward
	pub fn cudnnScaleTensor(handle: cudnnHandle_t,
			yDesc: cudnnTensorDescriptor_t,
			y: gpuMem_t,
			alpha: *const c_void) -> cudnnStatus_t; // alpha is in host mem
	
	pub fn cudnnActivationForward(handle: cudnnHandle_t,
			activationDesc: cudnnActivationDescriptor_t,
			alpha: *const c_void,
			xDesc: cudnnTensorDescriptor_t,
			x: gpuMem_t,
			beta: *const c_void,
			yDesc: cudnnTensorDescriptor_t,
			y: gpuMem_t) -> cudnnStatus_t;
	
	pub fn cudnnActivationBackward(handle: cudnnHandle_t,
			activationDesc: cudnnActivationDescriptor_t,
			alpha: *const c_void,
			yDesc: cudnnTensorDescriptor_t,
			y: gpuMem_t,
			dyDesc: cudnnTensorDescriptor_t,
			dy: gpuMem_t,
			xDesc: cudnnTensorDescriptor_t,
			x: gpuMem_t,
			beta: *const c_void,
			dxDesc: cudnnTensorDescriptor_t,
			dx: gpuMem_t) -> cudnnStatus_t;
	
	pub fn cudnnPoolingForward(handle: cudnnHandle_t,
			poolingDesc: cudnnPoolingDescriptor_t,
			alpha: *const c_void,
			xDesc: cudnnTensorDescriptor_t,
			x: gpuMem_t,
			beta: *const c_void,
			yDesc: cudnnTensorDescriptor_t,
			y: gpuMem_t) -> cudnnStatus_t;

	pub fn cudnnPoolingBackward(handle: cudnnHandle_t,
			poolingDesc: cudnnPoolingDescriptor_t,
			alpha: *const c_void,
			yDesc: cudnnTensorDescriptor_t,
			y: gpuMem_t,
			dyDesc: cudnnTensorDescriptor_t,
			dy: gpuMem_t,
			xDesc: cudnnTensorDescriptor_t,
			x: gpuMem_t,
			beta: *const c_void,
			dxDesc: cudnnTensorDescriptor_t,
			dx: gpuMem_t) -> cudnnStatus_t;
	
	pub fn cudnnConvolutionForward(handle: cudnnHandle_t,
			alpha: *const c_void,
			xDesc: cudnnTensorDescriptor_t,
			x: gpuMem_t,
			wDesc: cudnnFilterDescriptor_t,
			w: gpuMem_t,
			convDesc: cudnnConvolutionDescriptor_t,
			algo: cudnnConvolutionFwdAlgo_t,
			workspace: gpuMem_t,
			workSpaceSizeInBytes: size_t,
			beta: *const c_void,
			yDesc: cudnnTensorDescriptor_t,
			y: gpuMem_t) -> cudnnStatus_t;
	
	pub fn cudnnConvolutionBackwardData(handle: cudnnHandle_t,
			alpha: *const c_void,
			wDesc: cudnnFilterDescriptor_t,
			w: gpuMem_t,
			dyDesc: cudnnTensorDescriptor_t,
			dy: gpuMem_t,
			convDesc: cudnnConvolutionDescriptor_t,
			algo: cudnnConvolutionBwdDataAlgo_t,
			workSpace: gpuMem_t,
			workSpaceSizeInBytes: size_t,
			beta: *const c_void,
			dxDesc: cudnnTensorDescriptor_t,
			dx: gpuMem_t) -> cudnnStatus_t;
	
	pub fn cudnnConvolutionBackwardFilter(handle: cudnnHandle_t,
			alpha: *const c_void,
			xDesc: cudnnTensorDescriptor_t,
			x: gpuMem_t,
			dyDesc: cudnnTensorDescriptor_t,
			dy: gpuMem_t,
			convDesc: cudnnConvolutionDescriptor_t,
			algo: cudnnConvolutionBwdFilterAlgo_t,
			workSpace: gpuMem_t,
			workSpaceSizeInBytes: size_t,
			beta: *const c_void,
			dwDesc: cudnnFilterDescriptor_t,
			dw: gpuMem_t) -> cudnnStatus_t;
	
	pub fn cudnnSoftmaxForward(handle: cudnnHandle_t,
			algorithm: cudnnSoftmaxAlgorithm_t,
			mode: cudnnSoftmaxMode_t,
			alpha: *const c_void,
			xDesc: cudnnTensorDescriptor_t,
			x: gpuMem_t,
			beta: *const c_void,
			yDesc: cudnnTensorDescriptor_t,
			y: gpuMem_t) -> cudnnStatus_t;
	
	pub fn cudnnSoftmaxBackward(handle: cudnnHandle_t,
			algorithm: cudnnSoftmaxAlgorithm_t,
			mode: cudnnSoftmaxMode_t,
			alpha: *const c_void,
			yDesc: cudnnTensorDescriptor_t,
			yData: gpuMem_t,
			dyDesc: cudnnTensorDescriptor_t,
			dy: gpuMem_t,
			beta: *const c_void,
			dxDesc: cudnnTensorDescriptor_t,
			dx: gpuMem_t) -> cudnnStatus_t;
	
	pub fn cudnnOpTensor(handle: cudnnHandle_t,
			opTensorDesc: cudnnOpTensorDescriptor_t,
			alpha1: *const c_void,
			aDesc: cudnnTensorDescriptor_t,
			A: gpuMem_t,
			alpha2: *const c_void,
			bDesc: cudnnTensorDescriptor_t,
			B: gpuMem_t,
			beta: *const c_void,
			cDesc: cudnnTensorDescriptor_t,
			C: gpuMem_t) -> cudnnStatus_t;
	
	pub fn cudnnAddTensor(handle: cudnnHandle_t,
			alpha: *const c_void,
			aDesc: cudnnTensorDescriptor_t,
			A: gpuMem_t,
			beta: *const c_void,
			cDesc: cudnnTensorDescriptor_t,
			C: gpuMem_t) -> cudnnStatus_t;
	
	pub fn cudnnReduceTensor(handle: cudnnHandle_t,
			reduceTensorDesc: cudnnReduceTensorDescriptor_t,
			indices: gpuMem_t,
			indicesSizeInBytes: size_t,
			workspace: gpuMem_t,
			workspaceSizeInBytes: size_t,
			alpha: *const c_void,
			aDesc: cudnnTensorDescriptor_t,
			A: gpuMem_t,
			beta: *const c_void,
			cDesc: cudnnTensorDescriptor_t,
			C: gpuMem_t) -> cudnnStatus_t;
	
	pub fn cudnnRNNForwardTraining(handle: cudnnHandle_t,
			rnnDesc: cudnnRNNDescriptor_t,
			seqLength: c_int,
			
			xDesc: *const cudnnTensorDescriptor_t, x: gpuMem_t,
			hxDesc: cudnnTensorDescriptor_t, hx: gpuMem_t,
			cxDesc: cudnnTensorDescriptor_t, cx: gpuMem_t,
			wDesc: cudnnFilterDescriptor_t, w: gpuMem_t,
			yDesc: *const cudnnTensorDescriptor_t, y: gpuMem_t,
			hyDesc: cudnnTensorDescriptor_t, hy: gpuMem_t,
			cyDesc: cudnnTensorDescriptor_t, cy: gpuMem_t,
			
			workSpace: gpuMem_t,
			workSpaceSizeInBytes: size_t,
			
			reserveSpace: gpuMem_t,
			reserveSpaceSizeInBytes: size_t) -> cudnnStatus_t;
	
	pub fn cudnnRNNBackwardWeights(handle: cudnnHandle_t,
			rnnDesc: cudnnRNNDescriptor_t,
			seqLength: c_int,
			
			xDesc: *const cudnnTensorDescriptor_t, x: gpuMem_t,
			hxDesc: cudnnTensorDescriptor_t, hx: gpuMem_t,
			yDesc: *const cudnnTensorDescriptor_t, y: gpuMem_t,
			
			workSpace: gpuMem_t,
			workSpaceSizeInBytes: size_t,
			
			dwDesc: cudnnFilterDescriptor_t,
			dw: gpuMem_t,
			
			reserveSpace: gpuMem_t,
			reserveSpaceSizeInBytes: size_t) -> cudnnStatus_t;
	
	pub fn cudnnRNNForwardTrainingEx(handle: cudnnHandle_t,
			rnnDesc: cudnnRNNDescriptor_t,
			xDesc: cudnnRNNDataDescriptor_t, x: gpuMem_t,
			hxDesc: cudnnTensorDescriptor_t, hx: gpuMem_t,
			cxDesc: cudnnTensorDescriptor_t, cx: gpuMem_t,
			wDesc: cudnnFilterDescriptor_t, w: gpuMem_t,
			yDesc: cudnnRNNDataDescriptor_t, y: gpuMem_t,
			hyDesc: cudnnTensorDescriptor_t, hy: gpuMem_t,
			cyDesc: cudnnTensorDescriptor_t, cy: gpuMem_t,
			kDesc: cudnnRNNDataDescriptor_t, keys: gpuMem_t,
			cDesc: cudnnRNNDataDescriptor_t, cAttn: gpuMem_t,
			iDesc: cudnnRNNDataDescriptor_t, iAttn: gpuMem_t,
			qDesc: cudnnRNNDataDescriptor_t, queries: gpuMem_t,
			
			workSpace: gpuMem_t,
			workSpaceSizeInBytes: size_t,
			
			reserveSpace: gpuMem_t,
			reserveSpaceSizeInBytes: size_t) -> cudnnStatus_t;
	
	pub fn cudnnRNNForwardInferenceEx(handle: cudnnHandle_t,
			rnnDesc: cudnnRNNDescriptor_t,
			xDesc: cudnnRNNDataDescriptor_t, x: gpuMem_t,
			hxDesc: cudnnTensorDescriptor_t, hx: gpuMem_t,
			cxDesc: cudnnTensorDescriptor_t, cx: gpuMem_t,
			wDesc: cudnnFilterDescriptor_t, w: gpuMem_t,
			yDesc: cudnnRNNDataDescriptor_t, y: gpuMem_t,
			hyDesc: cudnnTensorDescriptor_t, hy: gpuMem_t,
			cyDesc: cudnnTensorDescriptor_t, cy: gpuMem_t,
			kDesc: cudnnRNNDataDescriptor_t, keys: gpuMem_t,
			cDesc: cudnnRNNDataDescriptor_t, cAttn: gpuMem_t,
			iDesc: cudnnRNNDataDescriptor_t, iAttn: gpuMem_t,
			qDesc: cudnnRNNDataDescriptor_t, queries: gpuMem_t,
			
			workSpace: gpuMem_t,
			workSpaceSizeInBytes: size_t) -> cudnnStatus_t;
	
	pub fn cudnnRNNBackwardWeightsEx(handle: cudnnHandle_t,
			rnnDesc: cudnnRNNDescriptor_t,
			xDesc: cudnnRNNDataDescriptor_t, x: gpuMem_t,
			hxDesc: cudnnTensorDescriptor_t, hx: gpuMem_t,
			yDesc: cudnnRNNDataDescriptor_t, y: gpuMem_t,
			
			workSpace: gpuMem_t,
			workSpaceSizeInBytes: size_t,
			
			dwDesc: cudnnFilterDescriptor_t, dw: gpuMem_t,
			
			reserveSpace: gpuMem_t, 
			reserveSpaceSizeInBytes: size_t) -> cudnnStatus_t;
	
	pub fn cudnnRNNBackwardDataEx(handle: cudnnHandle_t,
			runnDesc: cudnnRNNDescriptor_t,
			yDesc: cudnnRNNDataDescriptor_t, y: gpuMem_t,
			dyDesc: cudnnRNNDataDescriptor_t, dy: gpuMem_t,
			
			dcDesc: cudnnRNNDataDescriptor_t, dcAttn: gpuMem_t, // null?
			dhyDesc: cudnnTensorDescriptor_t, dhy: gpuMem_t,
			dcyDesc: cudnnTensorDescriptor_t, dcy: gpuMem_t,
			
			wDesc: cudnnFilterDescriptor_t, w: gpuMem_t,
			
			hxDesc: cudnnTensorDescriptor_t, hx: gpuMem_t,
			cxDesc: cudnnTensorDescriptor_t, cx: gpuMem_t,
			dxDesc: cudnnRNNDataDescriptor_t, dx: gpuMem_t,
			dhxDesc: cudnnTensorDescriptor_t, dhx: gpuMem_t,
			dcxDesc: cudnnTensorDescriptor_t, dcx: gpuMem_t,
			dkDesc: cudnnRNNDataDescriptor_t, dkeys: gpuMem_t, // null
			
			workSpace: gpuMem_t,
			workSpaceSizeInBytes: size_t,
			
			reserveSpace: gpuMem_t,
			reserveSpaceSizeInBytes: size_t) -> cudnnStatus_t;
}

// additional reference:
// https://github.com/rust-cuda/cuda-sys, accessed Feb 7, 2020

// /usr/local/cuda/include/driver_types.h: line 1031, accessed March 7, 2020
#[repr(C)]
pub enum cudaMemcpyKind {
	cudaMemcpyHostToHost = 0,
	cudaMemcpyHostToDevice = 1,
	cudaMemcpyDeviceToHost = 2,
	cudaMemcpyDeviceToDevice = 3,
	cudaMemcpyDefault = 4
}

///////////////////////////////////////////////////////////////////////////////////////////
// cudart functions
#[link(name = "cudart")]
extern "C" {
	// see: /usr/local/cuda/include/cuda_runtime_api.h; line 1895 (Feb 7, 2020)
	pub fn cudaSetDevice(device: c_int) -> cudaError_t;
	
	pub fn cudaMalloc(devPtr: *mut gpuMem_t, size: size_t) -> cudaError_t;
	pub fn cudaFree(devPtr: gpuMem_t) -> cudaError_t;
	
	pub fn cudaMemcpy(dst: *mut c_void, src: *const c_void, size: size_t, kind: cudaMemcpyKind) -> cudaError_t;
	pub fn cudaMemset(devPtr: gpuMem_t, value: c_int, bytes: size_t) -> cudaError_t;
	
	pub fn cudaMemGetInfo(free: *mut size_t, total: *mut size_t) -> cudaError_t;
	
	pub fn cudaDeviceSynchronize() -> cudaError_t;
}

///////////////////////////////////////////////////////////////////////////////////////////////
// custom kernels
#[link(name = "kernels")]
extern "C" {
	pub fn broadcast_across_channel_vals(dy: gpuMem_t, dx: gpuMem_t, n_imgs: size_t, n_channels: size_t, hw: size_t);
	pub fn broadcast_across_img_vals(dy: gpuMem_t, dx: gpuMem_t, vals_per_img: size_t, total_len: size_t);
	pub fn broadcast_across_all_vals(dy: gpuMem_t, dx: gpuMem_t, total_len: size_t);
	
	pub fn mat_mul_contract_inner_outer(N: size_t, K_outer: size_t, K_inner: size_t,
			M: size_t, C: gpuMem_t, B: gpuMem_t, A: gpuMem_t, beta: c_float);
	
	pub fn mat_mul_contract_inner_outer_f16(N: size_t, K_outer: size_t, K_inner: size_t,
			M: size_t, C: gpuMem_t, B: gpuMem_t, A: gpuMem_t, beta: c_float);
	
	pub fn shift_QR_pos_lleft_triangle(y: gpuMem_t, n_heads_imgs: size_t, n_time: size_t);
	pub fn shift_QR_pos_uright_triangle(dy: gpuMem_t, n_heads_imgs: size_t, n_time: size_t);
	
	pub fn mask_future_times_in_place(y: gpuMem_t, scale: f32, n_exemplars: size_t, n_time: size_t);
	pub fn mask_future_times(y: gpuMem_t, x: gpuMem_t, scale: f32, n_exemplars: size_t, n_time: size_t);
	
	
	pub fn mask_future_times_add(y: gpuMem_t, x1: gpuMem_t,
			x2: gpuMem_t, scale: f32,
			n_exemplars: size_t, n_time: size_t);
	
	pub fn transpose(dim0: size_t, dim1: size_t, dim2: size_t, dim3: size_t,
					sz_0: size_t, sz_1: size_t, sz_2: size_t, sz_3: size_t,
					x: gpuMem_t, y: gpuMem_t, beta: f32);
	
	pub fn pow_forward(x: gpuMem_t, alpha: c_float, y: gpuMem_t, len: size_t);
	pub fn pow_backward(x: gpuMem_t, alpha: c_float, dy: gpuMem_t, dx: gpuMem_t, len: size_t);
	
	pub fn dbias_plus_dy(dbias: gpuMem_t, dy: gpuMem_t, n_batches: size_t, bias_len: size_t);
	
	pub fn rms_update(alpha: c_float, eps: c_float, denom_eps: c_float,
			dw: gpuMem_t, weights_rms_tmp: gpuMem_t, w: gpuMem_t, len: size_t);
	
	pub fn adam_update(a_t: c_float, beta1: c_float, beta2: c_float, denom_eps: c_float,
			dw: gpuMem_t, m: gpuMem_t, v: gpuMem_t, w: gpuMem_t, len: size_t);
	
	pub fn adam_update_f16(a_t: c_float, beta1: c_float, beta2: c_float, denom_eps: c_float,
			dw: gpuMem_t, m: gpuMem_t, v: gpuMem_t, w: gpuMem_t, len: size_t);
	
	#[cfg(feature="titan_card_bypass_sgemm")]
	pub fn mat_mul_Bt_A(M: c_int, N: c_int, K: c_int, beta: c_float, C: gpuMem_t, B: gpuMem_t, A: gpuMem_t);
}

/////////////////////////////////////////////////////////////////////////////////////////
// Cublas
//
// /usr/include/cublas_v2.h, Accessed May 15, 2020
// /usr/include/cublas_api.h Accessed May 16, 2020

#[derive(PartialEq)]
#[repr(C)]
pub enum cublasStatus_t {
	CUBLAS_STATUS_SUCCESS = 0,
	CUBLAS_STATUS_NOT_INITIALIZED = 1,
	CUBLAS_STATUS_ALLOC_FAILED = 3,
	CUBLAS_STATUS_INVALID_VALUE = 7,
	CUBLAS_STATUS_ARCH_MISMATCH = 8,
	CUBLAS_STATUS_MAPPING_ERROR = 11,
	CUBLAS_STATUS_EXECUTION_FAILED = 13,
	CUBLAS_STATUS_INTERNAL_ERROR = 14,
	CUBLAS_STATUS_NOT_SUPPORTED = 15,
	CUBLAS_STATUS_LICENSE_ERROR = 16
}

#[repr(C)]
pub enum cublasOperation_t {
	CUBLAS_OP_N = 0,
	CUBLAS_OP_T = 1,
	CUBLAS_OP_C = 2,
	//CUBLAS_OP_HERMITAN = 2,
	CUBLAS_OP_CONJG = 3
}

// /usr/include/cublas_v2.h, Accessed May 15, 2020

#[link(name = "cublas")]
extern "C" {
	pub fn cublasCreate_v2(handle: *mut cublasHandle_t) -> cublasStatus_t;
	pub fn cublasDestroy_v2(handle: cublasHandle_t) -> cublasStatus_t;
	
	// https://docs.nvidia.com/cuda/cublas/index.html#cublas-lt-t-gt-gemmstridedbatched accessed May 14, 2020
	pub fn cublasSgemmStridedBatched(handle: cublasHandle_t,
						transa: cublasOperation_t,
						transb: cublasOperation_t,
						m: c_int, n: c_int, k: c_int,
						alpha: *const c_float, // by default, this is assumed to be on the host
						A: gpuMem_t,
						lda: c_int,
						strideA: c_longlong,
						B: gpuMem_t,
						ldb: c_int,
						strideB: c_longlong,
						beta: *const c_float,
						C: gpuMem_t,
						ldc: c_int,
						strideC: c_longlong,
						batch_count: c_int) -> cublasStatus_t;
	
	
	pub fn cublasHgemmStridedBatched(handle: cublasHandle_t,
						transa: cublasOperation_t,
						transb: cublasOperation_t,
						m: c_int, n: c_int, k: c_int,
						alpha: *const c_float, // by default, this is assumed to be on the host
						A: gpuMem_t,
						lda: c_int,
						strideA: c_longlong,
						B: gpuMem_t,
						ldb: c_int,
						strideB: c_longlong,
						beta: *const c_float,
						C: gpuMem_t,
						ldc: c_int,
						strideC: c_longlong,
						batch_count: c_int) -> cublasStatus_t;
	
	/* cublasStatus_t cublasSgemm(cublasHandle_t handle,
                           cublasOperation_t transa, cublasOperation_t transb,
                           int m, int n, int k,
                           const float           *alpha,
                           const float           *A, int lda,
                           const float           *B, int ldb,
                           const float           *beta,
                           float           *C, int ldc)
      */
	/*pub fn cublasSgemm(handle: cublasHandle_t,
					transa: cublasOperation_t,
					transb: cublasOperation_t,
					m: c_int, n: c_int, k: c_int,
					alpha: *const c_float,
					A: gpuMem_t,
					lda: c_int,
					B: gpuMem_t,
					ldb: c_int,
					beta: *const c_float,
					C: gpuMem_t,
					ldc: c_int) -> cublasStatus_t;*/
	// ^ fails with: (in einsum.rs, but works when an equivalent call with batch_count of 1 is given to cublasSgemmStridedBatched):
	// 	"** On entry to SGEMM  parameter number 1 had an illegal value"
}

