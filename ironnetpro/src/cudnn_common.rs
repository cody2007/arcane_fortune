// see cudnn/raw.rs
#![allow(non_camel_case_types)]

#[derive(Copy, Clone, PartialEq)]
#[repr(C)]
pub enum cudnnDataType_t {
	CUDNN_DATA_FLOAT = 0,
	CUDNN_DATA_DOUBLE = 1,
	CUDNN_DATA_HALF = 2,
	CUDNN_DATA_INT8 = 3,
	CUDNN_DATA_INT32 = 4,
	CUDNN_DATA_INT8x4 = 5,
	CUDNN_DATA_UINT8 = 6,
	CUDNN_DATA_UINT8x4 = 7,
	CUDNN_DATA_INT8x32 = 8
}

#[derive(PartialEq)]
#[repr(C)]
pub enum cudnnTensorFormat_t {
	CUDNN_TENSOR_NCHW = 0,
	CUDNN_TENSOR_NHWC = 1,
	CUDNN_TENSOR_NCHW_VECT_C = 2
}

#[derive(Copy, Clone)]
#[repr(C)]
pub enum cudnnSoftmaxAlgorithm_t {
	CUDNN_SOFTMAX_FAST = 0,
	CUDNN_SOFTMAX_ACCURATE = 1,
	CUDNN_SOFTMAX_LOG = 2
}

#[repr(C)]
#[derive(PartialEq)]
pub enum cudnnRNNDataLayout_t {
	CUDNN_RNN_DATA_LAYOUT_SEQ_MAJOR_UNPACKED = 0, // "padded, outer stride from one time-step to the next"
	CUDNN_RNN_DATA_LAYOUT_SEQ_MAJOR_PACKED = 1, // "sequence length sorted and packed as in basic RNN api"
	CUDNN_RNN_DATA_LAYOUT_BATCH_MAJOR_UNPACKED = 2 // "padded, outer stride from one batch to the next" (cudnn.h)
}

