#![allow(non_snake_case)]
use std::os::raw::{c_int, c_longlong, c_float};
use crate::model::Model;
use crate::data_wrappers::{MemWShape, size_t, prod};
use crate::cudnn::raw::*;
use crate::cudnn_common::cudnnDataType_t;
use super::{f32_to_f16};//, f16};

impl Model {
	pub fn einsum<Ct: MemWShape, Bt: MemWShape, At: MemWShape>(&self,
										C: &Ct,      C_dims: &[usize],
										B: &Bt,      B_dims: &[usize],
										A: &At,      A_dims: &[usize],
										N: c_int, K: c_int, M: c_int,
										batch_sz: c_int,
										beta: c_float, // beta = 1 => increment to C, beta = 0 => don't increment
										data_type: cudnnDataType_t) {
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
		
		let (C_shape, B_shape, A_shape) = (C.shape(), B.shape(), A.shape());
		
		macro_rules! handle_conditions{($fn: ident, $mat_mul_contract_inner_outer: ident, $type: ty, $one: ident, $beta: ident) => {
			let one_f = vec![$one];
			let beta_f = vec![$beta];
			
			/* ////////////////////////////////////////////////////////////////////////////
			   C[0,2] = B[0,1] * A[2,1]
				   transpose A
			*/			
			if C_dims == [0,2] && B_dims == [0,1] && A_dims == [2,1] {
				assert!(batch_sz == 1);
				assert!(prod(&C_shape) == (N*M), "C shape {:?} N: {} M: {}", C_shape, N, M);
				assert!(prod(&B_shape) == (N*K));
				assert!(prod(&A_shape) == (M*K));
				
				//println!("sgemm m {} n {} k {} T N", M,N,K);
				
				unsafe {$fn(self.handle.cublas_val,
									cublasOperation_t::CUBLAS_OP_T, // transa
									cublasOperation_t::CUBLAS_OP_N, // transb
									M, N,	K,
									one_f.as_ptr() as *const $type, // alpha
									
									A.mem(), // A
									K, // lda
									0,
									
									B.mem(), // B
									K, // ldb
									0,
									
									beta_f.as_ptr() as *const $type, // beta
									
									C.mem(), // C
									M, // ldc
									0,
									1
							)}.chk_err();
				//println!("fin");
			
			/* ////////////////////////////////////////////////////////////////////////////
			   C[0,1] = B[0,2] * A[2,1]
			*/
			}else if C_dims == [0,1] && B_dims == [0,2] && A_dims == [2,1] {
				assert!(batch_sz == 1);
				assert!(prod(&C_shape) == (N*M));
				assert!(prod(&B_shape) == (N*K));
				assert!(prod(&A_shape) == (K*M));
				
				//println!("sgemm m {} n {} k {} N N", M,N,K);
				
				unsafe {$fn(self.handle.cublas_val,
									cublasOperation_t::CUBLAS_OP_N, // transa
									cublasOperation_t::CUBLAS_OP_N, // transb
									M, N,	K,
									one_f.as_ptr() as *const $type, // alpha
									
									A.mem(), // A
									M, // lda
									0,
									
									B.mem(), // B
									K, // ldb
									0,
									
									beta_f.as_ptr() as *const $type, // beta
									
									C.mem(), // C
									M, // ldc
									0,
									1
							)}.chk_err();
				//println!("fin");
				
			/* ////////////////////////////////////////////////////////////////////////////
			   C[2,1] = B[0,2] * A[0,1]
				   transpose B
			*/
			}else if C_dims == [2,1] && B_dims == [0,2] && A_dims == [0,1] {
				assert!(batch_sz == 1);
				assert!(prod(&C_shape) == (N*M));
				assert!(prod(&B_shape) == (K*N));
				assert!(prod(&A_shape) == (K*M));
				
				//println!("sgemm m {} n {} k {} N T", M,N,K);
				#[cfg(feature="titan_card_bypass_sgemm")]
				{
					// 32-37 sec per 200 batches for a 5 layer transformer training seq lengths of 200, batch sz of 64, ff=1024, n_heads=8, vec_in=16*n_heads
					// 25-31 sec for the cublas sgemm function w/ the same model
					assert!(data_type == cudnnDataType_t::CUDNN_DATA_FLOAT);
					unsafe {mat_mul_Bt_A(M, N, K, beta, C.mem(), B.mem(), A.mem())};
				}
				
				
				// for some reason crashses on the Titan card (when run with single precision):
				//	../nptl/pthread_mutex_lock.c:81: __pthread_mutex_lock: Assertion `mutex->__data.__owner == 0' failed.
				//	Aborted (core dumped)
				// when the non batched gemm function is called the error is about argument 0 being invalid,
				// presumably the trans_a argument. switching the trans_a argument to transpose A doesn't
				// solve the issue.
				#[cfg(not(feature="titan_card_bypass_sgemm"))]
				unsafe {$fn(self.handle.cublas_val,
									cublasOperation_t::CUBLAS_OP_N, // transa
									cublasOperation_t::CUBLAS_OP_T, // transb
									M, N,	K,
									one_f.as_ptr() as *const $type, // alpha
									
									A.mem(), // A
									M, // lda
									0,
									
									B.mem(), // B
									N, // ldb
									0,
									
									beta_f.as_ptr() as *const $type, // beta
									
									C.mem(), // C
									M, // ldc
									0,
									1
							)}.chk_err();
				
				//println!("fin");
			
			/* ////////////////////////////////////////////////////////////////////////////
			   C[2,0,3] = B[0,1] * A[2,1,3]
				aka dim 2 is the batch of C & A (no striding between B)
				and then it can be thought of as std matrix multiplies
				(contraction: dim 1)
			*/
			}else if C_dims == [2,0,3] && B_dims == [0,1] && A_dims == [2,1,3] {
				assert!(prod(&C_shape) == (batch_sz*N*M));
				assert!(prod(&B_shape) == (N*K));
				assert!(prod(&A_shape) == (batch_sz*K*M));
				
				assert!((C_shape[0] == batch_sz && C_shape[1] == N && C_shape[2] == M) ||
					  (C_shape[0] == batch_sz && (C_shape[1]*C_shape[2]) == N && C_shape[3] == M),
						"{}, {}, {} != {}, {}, {}", C_shape[0], C_shape[1], C_shape[2], batch_sz, N, M);
				
				assert!((B_shape[0] == N && B_shape[1] == K) ||
					 ((B_shape[0]*B_shape[1]) == N && B_shape[2] == K),
						"{}, {} != {}, {}", B_shape[0], B_shape[1], N, K);
				
				assert!(A_shape[0] == batch_sz && A_shape[1] == K && A_shape[2] == M);
				
				unsafe {$fn(self.handle.cublas_val,
									cublasOperation_t::CUBLAS_OP_N, // transa
									cublasOperation_t::CUBLAS_OP_N, // transb
									M, N,	K,
									one_f.as_ptr() as *const $type, // alpha
									
									A.mem(), // A
									M, // lda
									(K*M) as c_longlong, // strideA
									
									B.mem(), // B
									K, // ldb
									0, // strideB
									
									beta_f.as_ptr() as *const $type, // beta
									
									C.mem(), // C
									M, // ldc
									(N*M) as c_longlong, // strideC
									batch_sz )}.chk_err();
			
			/* /////////////////////////////////////////////////////////////////////////////
			   C[2,1,3] = B[0,1] * A[2,0,3]
				aka dim 2 is the batch of C & A (no striding between B)
				B is transposed
				(contraction: dim 0)
			*/
			}else if C_dims == [2,1,3] && B_dims == [0,1] && A_dims == [2,0,3] {
				assert!(prod(&C_shape) == (batch_sz*N*M));
				assert!(prod(&B_shape) == (K*N));
				assert!(prod(&A_shape) == (batch_sz*K*M));
				
				assert!(C_shape[0] == batch_sz && C_shape[1] == N && C_shape[2] == M);
				
				assert!((B_shape[0] == K && B_shape[1] == N) ||
					 ((B_shape[0]*B_shape[1]) == K && B_shape[2] == N),
						"{}, {} != {}, {}", B_shape[0], B_shape[1], K, N);
				
				assert!((A_shape[0] == batch_sz && A_shape[1] == K && A_shape[2] == M) ||
					  (A_shape[0] == batch_sz && (A_shape[1]*A_shape[2]) == K && A_shape[3] == M),
						"{}, {} != {}, {}, {}", A_shape[0], A_shape[1], batch_sz, K, M);
				
				unsafe {$fn(self.handle.cublas_val,
									cublasOperation_t::CUBLAS_OP_N, // transa
									cublasOperation_t::CUBLAS_OP_T, // transb
									M, N,	K,
									one_f.as_ptr() as *const $type, // alpha
									
									A.mem(), // A
									M, // lda
									(K*M) as c_longlong, // strideA
									
									B.mem(), // B
									N, // ldb
									0, // strideB
									
									beta_f.as_ptr() as *const $type, // beta
									
									C.mem(), // C
									M, // ldc
									(N*M) as c_longlong, // strideC
									batch_sz )}.chk_err();
			
			/* /////////////////////////////////////////////////////////////////////
			  C[0,1] = B[2,0,3] * A[2,1,3]
				(contraction: dims 2 & 3 ---> custom kernel)
			*/
			}else if C_dims == [0,1] && B_dims == [2,0,3] && A_dims == [2,1,3] {
				assert!(batch_sz == 1);
				assert!(prod(&C_shape) == (N*M));
				assert!(prod(&B_shape) == (N*K)); // note: K is split (inner & outer dims around N: K_outer*N*K_inner)
				assert!(prod(&A_shape) == (M*K)); // "
				
				let K_outer = B_shape[0];
				let K_inner = *B_shape.last().unwrap();
				
				assert!((K_outer*K_inner) == K);
				assert!(*A_shape.last().unwrap() == K_inner);
				assert!(A_shape[0] == K_outer);
				
				unsafe {$mat_mul_contract_inner_outer(
						N as size_t,
						K_outer as size_t,
						K_inner as size_t,
						M as size_t,
						C.mem(), B.mem(), A.mem(), beta)};
			
			/* ///////////////////////////////////////////////////////////////////
			   C[0,1,3] = B[0,1,2] * A[0,3,2]
				aka dim 0 is the batch of C, B, A
				A is transposed
				(contraction: dim 2)
			*/
			}else if C_dims == [0,1,3] && B_dims == [0,1,2] && A_dims == [0,3,2] {
				assert!(batch_sz != 1);
				assert!(prod(&C_shape) == (batch_sz*N*M));
				assert!(prod(&B_shape) == (batch_sz*N*K));
				assert!(prod(&A_shape) == (batch_sz*M*K));
				
				assert!((C_shape[0] == batch_sz && C_shape[1] == N && C_shape[2] == M) ||
					  ((C_shape[0]*C_shape[1]) == batch_sz && C_shape[2] == N && C_shape[3] == M));
				
				assert!((B_shape[0] == batch_sz && B_shape[1] == N && B_shape[2] == K) ||
					  ((B_shape[0]*B_shape[1]) == batch_sz && B_shape[2] == N && B_shape[3] == K));
				
				assert!((A_shape[0] == batch_sz && A_shape[1] == M && A_shape[2] == K) ||
					  ((A_shape[0]*A_shape[1]) == batch_sz && A_shape[2] == M && A_shape[3] == K));
				
				unsafe {$fn(self.handle.cublas_val,
									cublasOperation_t::CUBLAS_OP_T, // transa
									cublasOperation_t::CUBLAS_OP_N, // transb
									M, N,	K,
									one_f.as_ptr() as *const $type, // alpha
									
									A.mem(), // A
									K, // lda
									(M*K) as c_longlong, // strideA
									
									B.mem(), // B
									K, // ldb
									(N*K) as c_longlong, // strideB
									
									beta_f.as_ptr() as *const $type, // beta
									
									C.mem(), // C
									M, // ldc
									(N*M) as c_longlong, // strideC
									batch_sz )}.chk_err();
			
			/* /////////////////////////////////////////////////////////////////
			   C[0,1,2] = B[0,1,3] * A[0,3,2]
				aka dim 0 is the batch of C, B, A
				(contraction: dim 3)
			*/
			}else if C_dims == [0,1,2] && B_dims == [0,1,3] && A_dims == [0,3,2] {
				assert!(batch_sz != 1);
				assert!(prod(&C_shape) == (batch_sz*N*M));
				assert!(prod(&B_shape) == (batch_sz*N*K));
				assert!(prod(&A_shape) == (batch_sz*K*M));
				
				assert!((C_shape[0] == batch_sz && C_shape[1] == N && C_shape[2] == M) ||
					  ((C_shape[0]*C_shape[1]) == batch_sz && C_shape[2] == N && C_shape[3] == M));
				
				assert!((B_shape[0] == batch_sz && B_shape[1] == N && B_shape[2] == K) ||
					  ((B_shape[0]*B_shape[1]) == batch_sz && B_shape[2] == N && B_shape[3] == K));
				
				assert!((A_shape[0] == batch_sz && A_shape[1] == K && A_shape[2] == M) ||
					  ((A_shape[0]*A_shape[1]) == batch_sz && A_shape[2] == K && A_shape[3] == M));
				
				unsafe {$fn(self.handle.cublas_val,
									cublasOperation_t::CUBLAS_OP_N, // transa
									cublasOperation_t::CUBLAS_OP_N, // transb
									M, N,	K,
									one_f.as_ptr() as *const $type, // alpha
									
									A.mem(), // A
									M, // lda
									(K*M) as c_longlong, // strideA
									
									B.mem(), // B
									K, // ldb
									(N*K) as c_longlong, // strideB
									
									beta_f.as_ptr() as *const $type, // beta
									
									C.mem(), // C
									M, // ldc
									(N*M) as c_longlong, // strideC
									batch_sz )}.chk_err();
			
			/* /////////////////////////////////////////////////////////////////
			   C[0,3,2] = B[0,1,3] * A[0,1,2]
				aka dim 0 is the batch of C, B, A
				B is transposed
				(contraction: dim 1)
			*/
			}else if C_dims == [0,3,2] && B_dims == [0,1,3] && A_dims == [0,1,2] {
				assert!(batch_sz != 1);
				assert!(prod(&C_shape) == (batch_sz*N*M));
				assert!(prod(&B_shape) == (batch_sz*K*N));
				assert!(prod(&A_shape) == (batch_sz*K*M));
				
				assert!((C_shape[0] == batch_sz && C_shape[1] == N && C_shape[2] == M) ||
					  ((C_shape[0]*C_shape[1]) == batch_sz && C_shape[2] == N && C_shape[3] == M));
				
				assert!((B_shape[0] == batch_sz && B_shape[1] == K && B_shape[2] == N) ||
					  ((B_shape[0]*B_shape[1]) == batch_sz && B_shape[2] == K && B_shape[3] == N));
				
				assert!((A_shape[0] == batch_sz && A_shape[1] == K && A_shape[2] == M) ||
					  ((A_shape[0]*A_shape[1]) == batch_sz && A_shape[2] == K && A_shape[3] == M));
				
				unsafe {$fn(self.handle.cublas_val,
									cublasOperation_t::CUBLAS_OP_N, // transa
									cublasOperation_t::CUBLAS_OP_T, // transb
									M, N,	K,
									one_f.as_ptr() as *const $type, // alpha
									
									A.mem(), // A
									M, // lda
									(K*M) as c_longlong, // strideA
									
									B.mem(), // B
									N, // ldb
									(K*N) as c_longlong, // strideB
									
									beta_f.as_ptr() as *const $type, // beta
									
									C.mem(), // C
									M, // ldc
									(N*M) as c_longlong, // strideC
									batch_sz )}.chk_err();
			
			/* /////////////////////////////////////////////////////////////////
			   C[0,1,3] = B[0,1,2] * A[0,2,3]
				aka dim 0 is the batch of C, B, A
				(contraction: dim 2)
			*/
			}else if C_dims == [0,1,3] && B_dims == [0,1,2] && A_dims == [0,2,3] {
				assert!(batch_sz != 1);
				assert!(prod(&C_shape) == (batch_sz*N*M));
				assert!(prod(&B_shape) == (batch_sz*N*K));
				assert!(prod(&A_shape) == (batch_sz*K*M));
				
				assert!((C_shape[0] == batch_sz && C_shape[1] == N && C_shape[2] == M) ||
					 ((C_shape[0]*C_shape[1]) == batch_sz && C_shape[2] == N && C_shape[3] == M) ||
					 (C_shape[0] == batch_sz && (C_shape[1]*C_shape[2]) == N && C_shape[3] == M));
				
				assert!((B_shape[0] == batch_sz && B_shape[1] == N && B_shape[2] == K) ||
					 ((B_shape[0]*B_shape[1]) == batch_sz && B_shape[2] == N && B_shape[3] == K) ||
					 (B_shape[0] == batch_sz && (B_shape[1]*B_shape[2]) == N && B_shape[3] == K));
				
				assert!((A_shape[0] == batch_sz && A_shape[1] == K && A_shape[2] == M) ||
					  ((A_shape[0]*A_shape[1]) == batch_sz && A_shape[2] == K && A_shape[3] == M));
				
				unsafe {$fn(self.handle.cublas_val,
									cublasOperation_t::CUBLAS_OP_N, // transa
									cublasOperation_t::CUBLAS_OP_N, // transb
									M, N,	K,
									one_f.as_ptr() as *const $type, // alpha
									
									A.mem(), // A
									M, // lda
									(K*M) as c_longlong, // strideA
									
									B.mem(), // B
									K, // ldb
									(N*K) as c_longlong, // strideB
									
									beta_f.as_ptr() as *const $type, // beta
									
									C.mem(), // C
									M, // ldc
									(N*M) as c_longlong, // strideC
									batch_sz )}.chk_err();
			
			/* /////////////////////////////////////////////////////////////////
			   C[0,1,2] = B[0,1,3] * A[0,2,3]
				aka dim 0 is the batch of C, B, A
				A is transposed
				(contraction: dim 3)
			*/
			}else if C_dims == [0,1,2] && B_dims == [0,1,3] && A_dims == [0,2,3] {
				assert!(batch_sz != 1);
				assert!(prod(&C_shape) == (batch_sz*N*M), "C_shape {:?} batch_sz {} N {} M {}", C_shape, batch_sz, N, M);
				assert!(prod(&B_shape) == (batch_sz*N*K));
				assert!(prod(&A_shape) == (batch_sz*M*K));
				
				assert!((C_shape[0] == batch_sz && C_shape[1] == N && C_shape[2] == M) ||
					  ((C_shape[0]*C_shape[1]) == batch_sz && C_shape[2] == N && C_shape[3] == M) ||
					  (C_shape[0] == batch_sz && (C_shape[1]*C_shape[2]) == N && C_shape[3] == M));
				
				assert!((B_shape[0] == batch_sz && B_shape[1] == N && B_shape[2] == K) ||
					  ((B_shape[0]*B_shape[1]) == batch_sz && B_shape[2] == N && B_shape[3] == K) ||
					  (B_shape[0] == batch_sz && (B_shape[1]*B_shape[2]) == N && B_shape[3] == K));
				
				assert!((A_shape[0] == batch_sz && A_shape[1] == M && A_shape[2] == K) ||
					  ((A_shape[0]*A_shape[1]) == batch_sz && A_shape[2] == M && A_shape[3] == K));
				
				unsafe {$fn(self.handle.cublas_val,
									cublasOperation_t::CUBLAS_OP_T, // transa
									cublasOperation_t::CUBLAS_OP_N, // transb
									M, N,	K,
									one_f.as_ptr() as *const $type, // alpha
									
									A.mem(), // A
									K, // lda
									(M*K) as c_longlong, // strideA
									
									B.mem(), // B
									K, // ldb
									(N*K) as c_longlong, // strideB
									
									beta_f.as_ptr() as *const $type, // beta
									
									C.mem(), // C
									M, // ldc
									(N*M) as c_longlong, // strideC
									batch_sz )}.chk_err();
			
			/* /////////////////////////////////////////////////////////////////
			   C[0,2,3] = B[0,1,2] * A[0,1,3]
				aka dim 0 is the batch of C, B, A
				B is transposed
				(contraction: dim 1)
			*/
			}else if C_dims == [0,2,3] && B_dims == [0,1,2] && A_dims == [0,1,3] {
				assert!(batch_sz != 1);
				assert!(prod(&C_shape) == (batch_sz*N*M));
				assert!(prod(&B_shape) == (batch_sz*K*N));
				assert!(prod(&A_shape) == (batch_sz*K*M));
				
				assert!((C_shape[0] == batch_sz && C_shape[1] == N && C_shape[2] == M) ||
					  ((C_shape[0]*C_shape[1]) == batch_sz && C_shape[2] == N && C_shape[3] == M));
				
				assert!((B_shape[0] == batch_sz && B_shape[1] == K && B_shape[2] == N) ||
					  ((B_shape[0]*B_shape[1]) == batch_sz && B_shape[2] == K && B_shape[3] == N) ||
					  (B_shape[0] == batch_sz && (B_shape[1]*B_shape[2]) == K && B_shape[3] == N));
				
				assert!((A_shape[0] == batch_sz && A_shape[1] == K && A_shape[2] == M) ||
					  ((A_shape[0]*A_shape[1]) == batch_sz && A_shape[2] == K && A_shape[3] == M) ||
					  (A_shape[0] == batch_sz && (A_shape[1]*A_shape[2]) == K && A_shape[3] == M));
				
				unsafe {$fn(self.handle.cublas_val,
									cublasOperation_t::CUBLAS_OP_N, // transa
									cublasOperation_t::CUBLAS_OP_T, // transb
									M, N,	K,
									one_f.as_ptr() as *const $type, // alpha
									
									A.mem(), // A
									M, // lda
									(K*M) as c_longlong, // strideA
									
									B.mem(), // B
									N, // ldb
									(K*N) as c_longlong, // strideB
									
									beta_f.as_ptr() as *const $type, // beta
									
									C.mem(), // C
									M, // ldc
									(N*M) as c_longlong, // strideC
									batch_sz )}.chk_err();
			}else{
				panic!("configuration not supported");
			}
			
			unsafe {cudaDeviceSynchronize()}.chk_err();
		};};
		
		match data_type {
			cudnnDataType_t::CUDNN_DATA_FLOAT => {
				let one = 1_f32;
				handle_conditions!(cublasSgemmStridedBatched, mat_mul_contract_inner_outer, c_float, one, beta);}
			cudnnDataType_t::CUDNN_DATA_HALF => {
				let beta = f32_to_f16(beta);
				let one = f32_to_f16(1_f32);
				handle_conditions!(cublasHgemmStridedBatched, mat_mul_contract_inner_outer_f16, c_float, one, beta);}
			_ => {panic!("unsupported data type");}
		}
	}
}

