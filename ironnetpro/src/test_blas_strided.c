// nvcc test_blas_strided.c -o test_blas_strided -lcublas
/*#include <string.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <stdlib.h>*/
#include <stdio.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <assert.h>

#define CHECK_CUDA_ERR {err = cudaGetLastError();if(err != cudaSuccess){\
	printf("CUDA error: %s, %s, %i\n",cudaGetErrorString(err),__FILE__,__LINE__);return 0;}}

int main() {
	cudaError_t err;
	
	cublasHandle_t blas_h;
	assert(cublasCreate(&blas_h) == CUBLAS_STATUS_SUCCESS);
	
	cudaSetDevice(0); CHECK_CUDA_ERR
	
	float alpha = 1.;
	float beta = 0.;
	/*cublasHgemmStridedBatched(cublasHandle_t handle,
                                  cublasOperation_t transa,
                                  cublasOperation_t transb,
                                  int m, int n, int k,
                                  const __half           *alpha,
                                  const __half           *A, int lda,
                                  long long int          strideA,
                                  const __half           *B, int ldb,
                                  long long int          strideB,
                                  const __half           *beta,
                                  __half                 *C, int ldc,
                                  long long int          strideC,
                                  int batchCount)*/
     /*
		   	https://devblogs.nvidia.com/cublas-strided-batched-matrix-multiply/ Accessed May 14, 2020
			https://docs.nvidia.com/cuda/cublas/#cublas-lt-t-gt-gemmbatched Accessed May 17, 2020
			
			N
			K = ldb
			M = lda = ldc
			
			for (int p = 0; p < batchCount; ++p) {
				for (int m = 0; m < M; ++m) {
					for (int n = 0; n < N; ++n) {
						for (int k = 0; k < K, ++k)
							c_mnp += B[p][n*ldb + k] * A[p][k*ldA + m];
						C[p][n*ldc + m] = c_mnp;
					}
				}
			}
		*/
	
	// C = B * A
	int L = 3;
	int M = 2;
	int BH = 4;
	int DH = 5;
	
	float *C, *B, *A;
	
	// C[L,BH,M] = B[L,BH,DH] * A[M,BH,DH]
	// C[0,1,2] = B[0,1,3] * A[2,1,3]
	//	dim 1 is the batch of C, B & A
	//	A is transposed
	//	(contraction: dim 3)
	cudaMalloc(&C, L*BH*M*sizeof(float)); CHECK_CUDA_ERR
	cudaMalloc(&B, L*BH*DH*sizeof(float)); CHECK_CUDA_ERR
	cudaMalloc(&A, M*BH*DH*sizeof(float)); CHECK_CUDA_ERR
	
	// C[N*B*M] = B[N*B*K] * A[M*B*K]
	int N = L;
	int K = DH;
	//int M = M;
	
      assert(cublasSgemStridedBatched(blas_h,
      			CUBLAS_OP_T, // transa
      			CUBLAS_OP_N, // transb
      			M, N, K,
      			&alpha,
      			A,
      			DH, // lda
      			// strideA
      			B,
      			DH, // ldb
      			// strideB
      			&beta,
      			C,
      			M, // ldc
      			// srideC
      			BH // batch count
				) == CUBLAS_STATUS_SUCCESS);
}

