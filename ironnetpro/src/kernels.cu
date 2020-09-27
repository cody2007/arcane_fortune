#include <stdio.h>
#include <cassert>
#include <limits>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

#define MAX_THREADS_PER_BLOCK 1024
#define MAX_BLOCKS 65535
#define THREAD_CAPACITY (MAX_BLOCKS*MAX_THREADS_PER_BLOCK)

#define CHECK_CUDA_ERR {err = cudaGetLastError();if(err != cudaSuccess){\
		printf("CUDA error: %s, %s, %i\n",cudaGetErrorString(err),__FILE__,__LINE__);return;}}

/////////////////////////////////////
// case of [n,1,1,1] being broadcast added to [n,c,h,w]
__global__ void broadcast_across_img_vals_kernel(const float * dy, float * dx, const size_t vals_per_img, const size_t total_len) {
	int ind = blockIdx.x*MAX_THREADS_PER_BLOCK + threadIdx.x;
	if(ind >= total_len) {return;}
	int img = ind / vals_per_img;
	dx[ind] += dy[img];
}

extern "C" void broadcast_across_img_vals(const float * dy, float * dx, const size_t vals_per_img, const size_t total_len) {
	int n_blocks = (int)ceil((double)(total_len)/MAX_THREADS_PER_BLOCK);
	//cudaError_t err = cudaDeviceSynchronize(); CHECK_CUDA_ERR
	
	broadcast_across_img_vals_kernel <<< n_blocks, MAX_THREADS_PER_BLOCK >>> (dy, dx, vals_per_img, total_len);
	//err = cudaDeviceSynchronize(); CHECK_CUDA_ERR
}

///////////////////////////////////
// case of [1,1,1,1] being broadcast added to [n,c,h,w]
__global__ void broadcast_across_all_vals_kernel(const float * dy, float * dx, const size_t total_len) {
	int ind = blockIdx.x*MAX_THREADS_PER_BLOCK + threadIdx.x;
	if(ind >= total_len) {return;}
	dx[ind] += dy[0];
}

extern "C" void broadcast_across_all_vals(const float * dy, float * dx, const size_t total_len) {
	int n_blocks = (int)ceil((double)(total_len)/MAX_THREADS_PER_BLOCK);
	
	broadcast_across_all_vals_kernel <<< n_blocks, MAX_THREADS_PER_BLOCK >>> (dy, dx, total_len);
}

////////////////////////////////////
// mat mul, contract inner & outer dims
//
// C[0,1] = (B[2,0,3] * A[2,1,3]) + beta*C[0,1]
//  	(contraction: dims 2 & 3 ---> custom kernel)
// N is dim 0
// K_outer is dim 2 (contraction dim)
// K_inner is dim 3 (contraction dim)
// M is dim 1
__global__ void mat_mul_contract_inner_outer_kernel(const size_t N, const size_t K_outer,
							   const size_t K_inner, const size_t M,
							   float * C, const float * B, const float * A,
							   const float beta) {
	int ind = blockIdx.x*MAX_THREADS_PER_BLOCK + threadIdx.x;
	if(ind >= (N*M)) {return;}
	
	// compute output for C[n,m]
	int n = ind / M;
	int m = ind % M;
	
	float val = 0.;
	for(int k_outer = 0; k_outer < K_outer; k_outer++) {
		for(int k_inner = 0; k_inner < K_inner; k_inner++) {
			val += B[k_outer*(N*K_inner) + n*K_inner + k_inner] * 
				 A[k_outer*(M*K_inner) + m*K_inner + k_inner];
		}
	}
	
	if(beta == 0.)
		C[ind] = val;
	else
		C[ind] = beta*C[ind] + val;
}

__global__ void mat_mul_contract_inner_outer_f16_kernel(const size_t N, const size_t K_outer,
							   const size_t K_inner, const size_t M,
							   __half2 * C, const __half2 * B, const __half2 * A,
							   const float beta) {
	/*int ind = blockIdx.x*MAX_THREADS_PER_BLOCK + threadIdx.x;
	if(ind >= (N*M)) {return;}
	
	// compute output for C[n,m]
	int n = ind / M;
	int m = ind % M;
	
	__half2 val = __float2half2_rn(0.);
	for(int k_outer = 0; k_outer < K_outer; k_outer++) {
		for(int k_inner = 0; k_inner < K_inner; k_inner++) {
			val += B[k_outer*(N*K_inner) + n*K_inner + k_inner],
				 A[k_outer*(M*K_inner) + m*K_inner + k_inner];
		}
	}
	
	if(beta == 0.)
		C[ind] = val;
	else
		C[ind] = __float2half2_rn(beta)*C[ind] + val;*/
}

extern "C" void mat_mul_contract_inner_outer(const size_t N, const size_t K_outer,
							   const size_t K_inner, const size_t M,
							   float * C, const float * B, const float * A,
							   const float beta) {
	int n_blocks = (int)ceil( (((double)N * (double)M)) / MAX_THREADS_PER_BLOCK);
	
	mat_mul_contract_inner_outer_kernel <<< n_blocks, MAX_THREADS_PER_BLOCK >>> (
			N, K_outer, K_inner, M, C, B, A, beta);
}

extern "C" void mat_mul_contract_inner_outer_f16(const size_t N, const size_t K_outer,
							   const size_t K_inner, const size_t M,
							   __half2 * C, const __half2 * B, const __half2 * A,
							   const float beta) {
	/*int n_blocks = (int)ceil( (((double)N * (double)M)) / MAX_THREADS_PER_BLOCK);
	
	mat_mul_contract_inner_outer_f16_kernel <<< n_blocks, MAX_THREADS_PER_BLOCK >>> (
			N, K_outer, K_inner, M, C, B, A, beta);*/
}

/////////////////////////////////////////
/*
      Shift Q*R positions to lower left triangle
      
      Y[h, img, :,:] = | P P P 0 |
	                 | P P 1 0 |
	                 | P 2 1 0 |
	                 | 3 2 1 0 |   matrix is: [time1, time2]
	     
	     where the numbers represent the rel distance (from R) that was multiplied with Q
	     and P represents garbage points (future time points) that should be removed.
	     We shift each row (for each img and head) to remove all `P`s, because only values of
	     t2 <= t1 (i.e., the lower left triangle) are used after the mask_future_times layer.
	     
	The shifting occurs in place and results in:
	
	Y[h, img, :,:] = | 0 U U U |
	                 | 1 0 U U |
	                 | 2 1 0 U |
	                 | 3 2 1 0 |   matrix is: [time1, time2]
	      
	      where U are undefined values (currently just the previous values)
	      
	below: n_heads_imgs = n_heads * n_imgs
	
	each kernel call shifts all the values in a row of the 2d matrix shown above
*/
__global__ void shift_QR_pos_lleft_triangle_kernel(float * y, size_t n_heads_imgs, size_t n_time) {
	int mat_offset = blockIdx.x*MAX_THREADS_PER_BLOCK + threadIdx.x;
	if(mat_offset >= (n_heads_imgs*n_time)) {return;}
	
	int row = mat_offset % n_time;
	int row_offset = mat_offset * n_time;
	
	int col_offset = n_time - 1 - row;
	
	for(int col = 0; col <= row; col++) {
		y[row_offset + col] = y[row_offset + col + col_offset];
	}
}

extern "C" void shift_QR_pos_lleft_triangle(float * y, size_t n_heads_imgs, size_t n_time) {
	int n_blocks = (int)ceil( ((double)(n_heads_imgs*n_time)) / MAX_THREADS_PER_BLOCK);
	
	shift_QR_pos_lleft_triangle_kernel <<< n_blocks, MAX_THREADS_PER_BLOCK >>> (
			y, n_heads_imgs, n_time);
}

/*
      Shift Q*R positions to upper right triangle (for backprop of the shift to the lower left triangle)
      
     dY[h, img, :,:] = | 0 U U U |
	                 | 1 0 U U |
	                 | 2 1 0 U |
	                 | 3 2 1 0 |   matrix is: [time1, time2]
	      
	      where U are undefined values (should be zero because the mask layer followed
	      by softmax should result in these values resulting in zero on the backward pass)
      
	The shifting occurs in place and results in:
	
     dY[h, img, :,:] = | Z Z Z 0 |
	                 | Z Z 1 0 |
	                 | Z 2 1 0 |
	                 | 3 2 1 0 |   matrix is: [time1, time2]
	     
	     where the numbers represent the rel distance (from R) that was multiplied with Q
	     and Z represents values set to zero.

	below: n_heads_imgs = n_heads * n_imgs
	
	each kernel call shifts all the values in a row of the 2d matrix shown above
*/
__global__ void shift_QR_pos_uright_triangle_kernel(float * dy, size_t n_heads_imgs, size_t n_time) {
	int mat_offset = blockIdx.x*MAX_THREADS_PER_BLOCK + threadIdx.x;
	if(mat_offset >= (n_heads_imgs*n_time)) {return;}
	
	int row = mat_offset % n_time;
	int row_offset = mat_offset * n_time;
	
	int col_offset = n_time - 1 - row;
	
	for(int col = row; col >= 0; col--) {
		dy[row_offset + col + col_offset] = dy[row_offset + col];
		if(col < col_offset)
			dy[row_offset + col] = 0.;
	}
}

extern "C" void shift_QR_pos_uright_triangle(float * dy, size_t n_heads_imgs, size_t n_time) {
	int n_blocks = (int)ceil( ((double)(n_heads_imgs*n_time)) / MAX_THREADS_PER_BLOCK);
	
	shift_QR_pos_uright_triangle_kernel <<< n_blocks, MAX_THREADS_PER_BLOCK >>> (
			dy, n_heads_imgs, n_time);
}


////////////////////////////////////////
// mask future times & scale
__global__ void mask_future_times_kernel(float * y, const float * x, const float scale,
		size_t n_exemplars, size_t n_time) {
	int ind = blockIdx.x*MAX_THREADS_PER_BLOCK + threadIdx.x;
	if(ind >= (n_exemplars*n_time*n_time)) {return;}
	
	int r = ind % (n_time * n_time);
	
	int time1 = r / n_time;
	int time2 = r % n_time;
	
	if(time2 > time1) {
		y[ind] = -INFINITY;
	}else{
		y[ind] = scale * x[ind];
	}
}

extern "C" void mask_future_times(float * y, const float * x, const float scale,
			size_t n_exemplars, size_t n_time) {
	int n_blocks = (int)ceil( (((double)n_exemplars * (double)n_time * (double)n_time)) 
					/ MAX_THREADS_PER_BLOCK);
	// Asserts floating point compatibility at compile time (https://stackoverflow.com/questions/20016600/negative-infinity, Accessed May 18, 2020)
	static_assert(std::numeric_limits<float>::is_iec559, "IEEE 754 required");
	mask_future_times_kernel <<< n_blocks, MAX_THREADS_PER_BLOCK >>> (y, x, scale, n_exemplars, n_time);
}

////////////////////////////////////////
// (1) add Q*K and Q*pos inputs, (2) scale, (3) mask future times
__global__ void mask_future_times_add_kernel(float * y, const float * x1, const float * x2,
		const float scale, size_t n_exemplars, size_t n_time) {
	int ind = blockIdx.x*MAX_THREADS_PER_BLOCK + threadIdx.x;
	if(ind >= (n_exemplars*n_time*n_time)) {return;}
	
	int r = ind % (n_time * n_time);
	
	int time1 = r / n_time;
	int time2 = r % n_time;
	
	if(time2 > time1) {
		y[ind] = -INFINITY;
	}else{
		y[ind] = scale * (x1[ind] + x2[ind]);
	}
}

extern "C" void mask_future_times_add(float * y, const float * x1, const float * x2,
			const float scale, size_t n_exemplars, size_t n_time) {
	int n_blocks = (int)ceil( (((double)n_exemplars * (double)n_time * (double)n_time)) 
					/ MAX_THREADS_PER_BLOCK);
	
	// Asserts floating point compatibility at compile time (https://stackoverflow.com/questions/20016600/negative-infinity, Accessed May 18, 2020)
	static_assert(std::numeric_limits<float>::is_iec559, "IEEE 754 required");
	mask_future_times_add_kernel <<< n_blocks, MAX_THREADS_PER_BLOCK >>> (y, x1, x2, scale, n_exemplars, n_time);
}

//////////////////////////////////////
// transpose x to y
//
// dim_X are dimensions that y should be transformed to (some reordering of [0,1,2,3],
// sz_X should be the number of elements in each of the dimensions (of x)

#define COMPUTE_INDS \
	int i_0 = x_ind / (sz_1*sz_2*sz_3); \
	int   r = x_ind % (sz_1*sz_2*sz_3); \
	\
	int i_1 = r / (sz_2*sz_3); \
	r      %=     (sz_2*sz_3); \
	\
	int i_2 = r / sz_3; \
	int i_3 = r % sz_3; \
	
// [0,1,2,3] -> [1,2,0,3]
__global__ void transpose_1203_kernel(size_t sz_0, size_t sz_1, size_t sz_2, size_t sz_3, size_t total_sz,
					    	  const float * x, float * y, float beta) {
	int x_ind = blockIdx.x*MAX_THREADS_PER_BLOCK + threadIdx.x;
	if(x_ind >= total_sz) {return;}
	
	COMPUTE_INDS
	
	int y_ind = i_1 * (sz_2*sz_0*sz_3) + \
			i_2 * (     sz_0*sz_3) + \
			i_0 *            sz_3  + \
			i_3;
	
	if(beta == 0)
		y[y_ind] = x[x_ind];
	else
		y[y_ind] = x[x_ind] + (beta*y[y_ind]);

}

// [0,1,2,3] -> [2,0,1,3]
__global__ void transpose_2013_kernel(size_t sz_0, size_t sz_1, size_t sz_2, size_t sz_3, size_t total_sz,
					        const float * x, float * y, float beta) {
	int x_ind = blockIdx.x*MAX_THREADS_PER_BLOCK + threadIdx.x;
	if(x_ind >= total_sz) {return;}
	
	COMPUTE_INDS
	
	int y_ind = i_2 * (sz_0*sz_1*sz_3) +
			i_0 * (     sz_1*sz_3) +
			i_1 *            sz_3  +
			i_3;
	
	if(beta == 0)
		y[y_ind] = x[x_ind];
	else
		y[y_ind] = x[x_ind] + (beta*y[y_ind]);
}

extern "C" void transpose(size_t dim_0, size_t dim_1, size_t dim_2, size_t dim_3,
					    size_t sz_0, size_t sz_1, size_t sz_2, size_t sz_3,
					    const float * x, float * y, float beta) {
	size_t total_sz = sz_0 * sz_1 * sz_2 * sz_3;
	
	int n_blocks = (int)ceil( ((double)total_sz) / MAX_THREADS_PER_BLOCK );
	
	// [0,1,2,3] -> [1,2,0,3]
	if(dim_0 == 1 && dim_1 == 2 && dim_2 == 0 && dim_3 == 3) {
		transpose_1203_kernel <<< n_blocks, MAX_THREADS_PER_BLOCK >>> (sz_0, sz_1, sz_2, sz_3, total_sz, x, y, beta);
		
	// [0,1,2,3] -> [2,0,1,3]
	}else if (dim_0 == 2 && dim_1 == 0 && dim_2 == 1 && dim_3 == 3) {
		transpose_2013_kernel <<< n_blocks, MAX_THREADS_PER_BLOCK >>> (sz_0, sz_1, sz_2, sz_3, total_sz, x, y, beta);
	
	// unsupported
	}else assert(false);
}


/////////////// pow: x^a
__global__ void pow_forward_kernel(const float * x, const float alpha, float * y, const size_t len) {
	int ind = blockIdx.x*MAX_THREADS_PER_BLOCK + threadIdx.x;
	if(ind >= len) {return;}
	// powf returns nan if x is negative, so if alpha is an integer, put in the abs value of x
	if(ceil(alpha) == alpha) {
		y[ind] = powf(abs(x[ind]), alpha);
	}else{
		y[ind] = powf(x[ind], alpha);
	}
	//printf("%f %f %f\n", x[ind], alpha, y[ind]);
}

__global__ void pow_backward_kernel(const float * x, const float alpha, const float * dy, float * dx, const size_t len) {
	int ind = blockIdx.x*MAX_THREADS_PER_BLOCK + threadIdx.x;
	if(ind >= len) {return;}
	
	// powf returns nan if x is negative, so if alpha is an integer, put in the abs value of x
	float pow_res;
	if(ceil(alpha) == alpha) {
		if(x[ind] < 0.) {
			pow_res = -powf(abs(x[ind]), alpha - 1.);
		}else{
			pow_res = powf(x[ind], alpha - 1.);
		}
	}else{
		pow_res = powf(x[ind], alpha - 1.);
	}

	dx[ind] += alpha * dy[ind] * pow_res;
	//printf("%f %f\n", alpha, dx[ind]);
}

extern "C" void pow_forward(const float * x, const float alpha, float * y, const size_t len) {
	int n_blocks = (int)ceil((double)(len)/MAX_THREADS_PER_BLOCK);
	//printf("alpha %f len %i n_blocks %i\n", alpha, len, n_blocks);
	//cudaError_t err = cudaDeviceSynchronize(); CHECK_CUDA_ERR

	pow_forward_kernel <<< n_blocks, MAX_THREADS_PER_BLOCK >>> (x, alpha, y, len);
	//err = cudaDeviceSynchronize(); CHECK_CUDA_ERR
}

extern "C" void pow_backward(const float * x, const float alpha, const float * dy, float * dx, const size_t len) {
	int n_blocks = (int)ceil((double)(len)/MAX_THREADS_PER_BLOCK);
	//cudaError_t err = cudaDeviceSynchronize(); CHECK_CUDA_ERR

	pow_backward_kernel <<< n_blocks, MAX_THREADS_PER_BLOCK >>> (x, alpha, dy, dx, len);
	//err = cudaDeviceSynchronize(); CHECK_CUDA_ERR
}

////////////// dbias += dy.sum(0),
//
//	where dbias: [1,1,height,width], and
//	`n_batches` is size of dim 0
//	`bias_sz` = height*width
__global__ void dbias_plus_dy_kernel(float * dbias, const float * dy,
						size_t n_batches, size_t bias_sz) {
	int ind = blockIdx.x*MAX_THREADS_PER_BLOCK + threadIdx.x;
	if(ind >= bias_sz) {return;}
	
	float sum = 0.;
	for(int batch = 0; batch < n_batches; batch++) {
		sum += dy[batch*bias_sz + ind];
	}
	
	dbias[ind] += sum;
}

extern "C" void dbias_plus_dy(float * dbias, const float * dy, size_t n_batches, size_t bias_sz) {
	int n_blocks = (int)ceil((double)(bias_sz)/MAX_THREADS_PER_BLOCK);
	
	dbias_plus_dy_kernel <<< n_blocks, MAX_THREADS_PER_BLOCK >>> (dbias, dy, n_batches, bias_sz);
}

/////////////// rms update
//
// weights_rms_tmp = alpha*weights_rms_tmp + (1-alpha)*dw^2
// w += eps*dw/(sqrt(weights_rms_tmp) + denom_eps)

__global__ void rms_update_kernel(
		const float alpha,
		const float eps,
		const float denom_eps,
		const float * dw, 
		float * weights_rms_tmp,
		float * w,
		const size_t len) {
	int ind = blockIdx.x*MAX_THREADS_PER_BLOCK + threadIdx.x;
	if(ind >= len) {return;}
	
	weights_rms_tmp[ind] = alpha*weights_rms_tmp[ind] + (1.-alpha)*(dw[ind]*dw[ind]);
	w[ind] += eps*dw[ind] / (sqrt(weights_rms_tmp[ind]) + denom_eps);
}

extern "C" void rms_update(
		const float alpha,
		const float eps,
		const float denom_eps,
		const float * dw, 
		float * weights_rms_tmp,
		float * w,
		const size_t len) {
	int n_blocks = (int)ceil((double)(len)/MAX_THREADS_PER_BLOCK);
	
	rms_update_kernel <<< n_blocks, MAX_THREADS_PER_BLOCK >>> (
			alpha, eps, denom_eps, dw, weights_rms_tmp, w, len);
}

/////////////// adam update
//
// m = beta1*m + (1 - beta1)*dw
// v = beta2*v + (1 - beta2)*(dw^2)
// w += a_t*m/(sqrt(v) + denom_eps)
__global__ void adam_update_kernel(
			const float a_t,
			const float beta1,
			const float beta2,
			const float denom_eps,
			const float * dw,
			float * m,
			float * v,
			float * w,
			const size_t len) {
	int ind = blockIdx.x*MAX_THREADS_PER_BLOCK + threadIdx.x;
	if(ind >= len) {return;}
	
	m[ind] = beta1 * m[ind] + (1 - beta1) * dw[ind];
	v[ind] = beta2 * v[ind] + (1 - beta2) * (dw[ind] * dw[ind]);
	
	w[ind] += a_t * m[ind] / (sqrtf(v[ind]) + denom_eps);
}

__global__ void adam_update_f16_kernel(
			const float a_t,
			const float beta1,
			const float beta2,
			const float denom_eps,
			const __half * dw,
			float * m,
			float * v,
			__half * w,
			const size_t len) {
	/*int ind = blockIdx.x*MAX_THREADS_PER_BLOCK + threadIdx.x;
	if(ind >= len) {return;}
	
	m[ind] = beta1 * m[ind] + (1 - beta1) * __half2float(dw[ind]);
	v[ind] = beta2 * v[ind] + (1 - beta2) * (__half2float(dw[ind]) * __half2float(dw[ind]));
	
	w[ind] += a_t * m[ind] / (sqrtf(v[ind]) + denom_eps);*/
}

extern "C" void adam_update(
			const float a_t,
			const float beta1,
			const float beta2,
			const float denom_eps,
			const float * dw,
			float * m,
			float * v,
			float * w,
			const size_t len) {
	int n_blocks = (int)ceil((double)(len)/MAX_THREADS_PER_BLOCK);
	
	adam_update_kernel <<< n_blocks, MAX_THREADS_PER_BLOCK >>> (
			a_t, beta1, beta2, denom_eps, dw, m, v, w, len);
}

extern "C" void adam_update_f16(
			const float a_t,
			const float beta1,
			const float beta2,
			const float denom_eps,
			const __half * dw,
			float * m,
			float * v,
			__half * w,
			const size_t len) {
	/*int n_blocks = (int)ceil((double)(len)/MAX_THREADS_PER_BLOCK);
	
	adam_update_f16_kernel <<< n_blocks, MAX_THREADS_PER_BLOCK >>> (
			a_t, beta1, beta2, denom_eps, dw, m, v, w, len);*/
}

