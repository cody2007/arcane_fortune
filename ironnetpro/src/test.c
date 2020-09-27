/*#include <string.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <stdlib.h>*/
#include <stdio.h>
#include <cuda_runtime.h>
#include <cudnn.h>

#define CHECK_CUDA_ERR {err = cudaGetLastError();if(err != cudaSuccess){\
	printf("CUDA error: %s, %s, %i\n",cudaGetErrorString(err),__FILE__,__LINE__);return 0;}}

#define CHECK_CUDNN {if(err != CUDNN_STATUS_SUCCESS) {printf("error: %i\n", __LINE__); return 0;}}

int main() {
	cudaError_t err;
	cudnnStatus_t cerr;
	
	cudaSetDevice(0); CHECK_CUDA_ERR
	
	cudnnHandle_t handle;
	err = cudnnCreate(&handle); CHECK_CUDNN
	
	cudnnActivationDescriptor_t activationDesc;
	cerr = cudnnCreateActivationDescriptor(&activationDesc); CHECK_CUDNN
	
	cudnnTensorDescriptor_t xDesc, yDesc;
	
	cerr = cudnnCreateTensorDescriptor(&xDesc);
	cerr = cudnnCreateTensorDescriptor(&yDesc);
	
	cerr = cudnnSetTensor4dDescriptor(xDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 2,3,4,5); CHECK_CUDNN
	cerr = cudnnSetTensor4dDescriptor(yDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 2,3,4,5); CHECK_CUDNN
	
	void *x, *y;
	
	cudaMalloc(&x, 2*3*4*5*sizeof(float)); CHECK_CUDA_ERR
	cudaMalloc(&y, 2*3*4*5*sizeof(float)); CHECK_CUDA_ERR
	
	cerr = cudnnSetActivationDescriptor(activationDesc,
			CUDNN_ACTIVATION_RELU, CUDNN_PROPAGATE_NAN, 0.); CHECK_CUDNN
	
	float alpha = 1.;
	float beta = 0.;
	
	cudnnActivationForward(handle, activationDesc, &alpha, xDesc, x, &beta, yDesc, y); CHECK_CUDNN
}

