#include <assert.h>
#include "kernels.h"
#include "gpu.h"
#include "snudnn.h"

//#define CDIV(a, b) (((((a) + (b) - 1)) / (b)) * (b))
#define CDIV(a, b) ((((a) + (b) - 1)) / (b))

#define TILE_DIM 16

__global__ void transpose_matrix_2d(float *in, float *out, int M, int N)
{
	// (M x N) -> (N x M)
	int ti = threadIdx.y;
	int tj = threadIdx.x;
	int bi = blockIdx.y;
	int bj = blockIdx.x;
	int i = bi * TILE_DIM + ti;
	int j = bj * TILE_DIM + tj;

	/*
	int i_out = (bj * TILE_DIM + ti);
	int j_out = (bi * TILE_DIM + tj);
	   __shared__ float block_g[TILE_DIM][TILE_DIM+1];

	//block_g[tj][ti] = (i < M && j < N) ? in[i * N + j] : 0.0f;
	block_g[ti][tj] = in[i*N+j];
	__syncthreads();
	
	if (i_out < N && j_out < M) {
		out[i_out * M + j_out] = block_g[tj][ti];
	}
	*/


	if (j >= N || i >= M) return;
	out[j * M + i] = in[i * N + j];
}

__global__ void transpose_matrix_3d_02(float *in, float *out, int M, int N, int K)
{
	int i = blockIdx.z * blockDim.z + threadIdx.z;
	int j = blockIdx.y * blockDim.y + threadIdx.y;
	int k = blockIdx.x * blockDim.x + threadIdx.x;

	if (i >= M || j >= N || k >= K)
		return;
	out[((k * N) + j) * M + i] = in[((i * N) + j) * K + k];
}

__global__ void transpose_matrix_3d_12(float *in, float *out, int M, int N, int K)
{
	int i = blockIdx.z * blockDim.z + threadIdx.z;
	int j = blockIdx.y * blockDim.y + threadIdx.y;
	int k = blockIdx.x * blockDim.x + threadIdx.x;

	if (i >= M || j >= N || k >= K)
		return;
	out[((i * K) + k) * N + j] = in[((i * N) + j) * K + k];
}

__global__ void transpose_matrix_3d_01(float *in, float *out, int M, int N, int K)
{
	int i = blockIdx.z * blockDim.z + threadIdx.z;
	int j = blockIdx.y * blockDim.y + threadIdx.y;
	int k = blockIdx.x * blockDim.x + threadIdx.x;

	if (i >= M || j >= N || k >= K)
		return;
	out[((j * M) + i) * K + k] = in[((i * N) + j) * K + k];

	//int idx = offset0 + in_1 * stride1 + offset1 + in_2 * stride2 + offset3;

	/*
	// d1 < d2
	int tid = blockIdx.x * blockDim.x + threadIdx.x;

	if (tid >= nelem) return;

	// 12, 4, 1
	// 0...48
	int sizes[3] = { nelem / S1, S1 / S2, S2 / S3 };
	int indices[3] = { tid / S1, (tid / S2) % sizes[1], tid % sizes[2] };


	int tmp;
	tmp = indices[d1]; indices[d1] = indices[d2]; indices[d2] = tmp;
	tmp = sizes[d1]; sizes[d1] = sizes[d2]; sizes[d2] = tmp;

	int outidx = (indices[0] * sizes[1] + indices[1]) * sizes[2] + indices[2];
	out[outidx] = in[tid];
	*/
}

__global__ void transpose_matrix_4d(float *in, float *out, int d1, int d2, int S1, int S2, int S3, int S4, int nelem)
{
	// d1 < d2
	int tid = blockIdx.x * blockDim.x + threadIdx.x;

	if (tid >= nelem) return;

	int sizes[4] = { nelem/S1, S1/S2, S2/S3, S3/S4 };
	int indices[4] = { tid / S1, (tid / S2) % sizes[1], (tid / S3) % sizes[2], tid % sizes[3] };

	int tmp;
	tmp = indices[d1]; indices[d1] = indices[d2]; indices[d2] = tmp;
	tmp = sizes[d1]; sizes[d1] = sizes[d2]; sizes[d2] = tmp;

	int outidx = ((indices[0] * sizes[1] + indices[1]) * sizes[2] + indices[2]) * sizes[3] + indices[3];
	out[outidx] = in[tid];
}


EXTERNC void cuda_transpose_matrix(float *in, float *out, int i, int j, int nelem, int dim, int* strides)
{
	assert( i != j );
	int max = i > j ? i : j;
	int min = i < j ? i : j;

	switch (dim) {
		case 2:
			{
				int M = nelem/strides[0];
				int N = strides[0];
				dim3 nThreads( TILE_DIM, TILE_DIM );
				dim3 nBlocks( CDIV(N, nThreads.x), CDIV(M, nThreads.y) );
				transpose_matrix_2d<<< nBlocks, nThreads >>>(in, out, M, N);
			}
			//transpose_matrix_4d<<<CDIV(nelem, 128), 128>>>(in, out, min, max, strides[0], strides[1], 1, 1, nelem);
			break;
		case 3:
			/*
			{
				int M = nelem/strides[0];
				int N = strides[0]/strides[1];
				int K = strides[1];

				if (min == 0 && max == 1) {
					dim3 nThreads( 8, 8, 8 );
					dim3 nBlocks( CDIV(K, nThreads.x), CDIV(N, nThreads.y), CDIV(M, nThreads.z) );
					transpose_matrix_3d_01<<< nBlocks, nThreads >>>(in, out, M, N, K);
				} else if (min == 0 && max == 2) {
					dim3 nThreads( 8, 8, 8 );
					dim3 nBlocks( CDIV(K, nThreads.x), CDIV(N, nThreads.y), CDIV(M, nThreads.z) );
					transpose_matrix_3d_02<<< nBlocks, nThreads >>>(in, out, M, N, K);
				} else {
					dim3 nThreads( 8, 8, 8 );
					dim3 nBlocks( CDIV(K, nThreads.x), CDIV(N, nThreads.y), CDIV(M, nThreads.z) );
					transpose_matrix_3d_12<<< nBlocks, nThreads >>>(in, out, M, N, K);
				}
			}
			*/
			transpose_matrix_4d<<<CDIV(nelem, 128), 128>>>(in, out, min, max, strides[0], strides[1], strides[2], 1, nelem);
			break;
		case 4:
			transpose_matrix_4d<<<CDIV(nelem, 128), 128>>>(in, out, min, max, strides[0], strides[1], strides[2], strides[3], nelem);
			break;
		default:
			printf("Not reach heere");
			exit(1);
	}
	chkCUDA(cudaGetLastError());
}

__global__ void gather_2d(const float *in, float *out, const int *index, int N, int M)
{
	int tid_x = blockIdx.x * blockDim.x + threadIdx.x;
	if (tid_x >= N) return;

	int idx = tid_x * M + index[tid_x];

	out[tid_x] = in[idx];
}

EXTERNC void cuda_gather_2d(const float *in, float *out, const int *index, int N, int M)
{
	// in: N x M
	// out: N
	// 0 <= index[i] < M for all i
	gather_2d<<<CDIV(N, 128), 128>>>(in, out, index, N, M);
	chkCUDA(cudaGetLastError());
}

__global__ void softmax_dy(const int *index, float *dy, int N, int M)
{
	int tid_x = blockIdx.x * blockDim.x + threadIdx.x;
	if (tid_x >= N) return;
	int idx = tid_x * M + index[tid_x];
	dy[idx] = - 1.0f / (float)N;
}

EXTERNC void cuda_softmax_dy(const int *index, float *dy, int N, int M)
{
	// dy: N x M
	// index: N
	// 0 <= index[i] < M for all i
	softmax_dy<<<CDIV(N, 128), 128>>>(index, dy, N, M);
	chkCUDA(cudaGetLastError());
}

__global__ void concat_2d_0(
		const float *x, const float *y, float *z,
		int sizeX0, int sizeX1,
		int sizeY0, int sizeY1,
		int sizeZ0, int sizeZ1)
{
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	int N = sizeZ0 * sizeZ1;
	if (tid >= N) return;

	int z0 = tid / sizeZ1;

	const float *buf;
	int idx;
	buf = (z0 < sizeX0) ? x : y;
	idx = (z0 < sizeX0) ? (tid) : (tid - sizeX0 * sizeX1);
	z[tid] = buf[idx];
}

__global__ void concat_2d_1(
		const float *x, const float *y, float *z,
		int sizeX0, int sizeX1,
		int sizeY0, int sizeY1,
		int sizeZ0, int sizeZ1)
{
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	int N = sizeZ0 * sizeZ1;
	if (tid >= N) return;

	int z0 = tid / sizeZ1;
	int z1 = tid % sizeZ1;

	const float *buf;
	int idx;
	buf = (z1 < sizeX1) ? x : y;
	int offset = z0;
	offset = (z1 < sizeX1) ? z0 : z0 + 1;
	int stride = (z1 < sizeX1) ? (sizeY1) : (sizeX1);
	idx = tid - offset * stride;
	z[tid] = buf[idx];
}

__global__ void concat_2d_1_same_8(
		const float *x, const float *y, float *z,
		int sizeX0, int sizeX1,
		int sizeY0, int sizeY1,
		int N)
{
	int tid0 = (blockIdx.x * blockDim.x + threadIdx.x) * 8;
	if (tid0 >= N) return;

	int offset1 = sizeX1/8;
	int tid1 = tid0 + offset1;
	int tid2 = tid1 + offset1;
	int tid3 = tid2 + offset1;
	int tid4 = tid3 + offset1;
	int tid5 = tid4 + offset1;
	int tid6 = tid5 + offset1;
	int tid7 = tid6 + offset1;

	float xv0 = x[tid0];   
	float xv1 = x[tid1];   
	float xv2 = x[tid2];   
	float xv3 = x[tid3];   
	float xv4 = x[tid4];   
	float xv5 = x[tid5];   
	float xv6 = x[tid6];   
	float xv7 = x[tid7];   
	float yv0 = y[tid0];
	float yv1 = y[tid1];
	float yv2 = y[tid2];
	float yv3 = y[tid3];
	float yv4 = y[tid4];
	float yv5 = y[tid5];
	float yv6 = y[tid6];
	float yv7 = y[tid7];
	
	z[tid0]   = xv0;
	z[tid1]   = xv1;
	z[tid2]   = xv2;
	z[tid3]   = xv3;
	z[tid4]   = xv4;
	z[tid5]   = xv5;
	z[tid6]   = xv6;
	z[tid7]   = xv7;

	z[tid0+N] = yv0;
	z[tid1+N] = yv1;
	z[tid2+N] = yv2;
	z[tid3+N] = yv3;
	z[tid4+N] = yv4;
	z[tid5+N] = yv5;
	z[tid6+N] = yv6;
	z[tid7+N] = yv7;
}

__global__ void concat_2d_1_same_32(
		const float *x, const float *y, float *z,
		int sizeX0, int sizeX1,
		int sizeY0, int sizeY1,
		int N)
{
	int tid0 = (blockIdx.x * blockDim.x + threadIdx.x) * 32;
	if (tid0 >= N) return;

	int offset1 = sizeX1/32;
	int tid1 = tid0 + offset1;
	int tid2 = tid1 + offset1;
	int tid3 = tid2 + offset1;
	int tid4 = tid3 + offset1;
	int tid5 = tid4 + offset1;
	int tid6 = tid5 + offset1;
	int tid7 = tid6 + offset1;
	int tid8 = tid7 + offset1;
	int tid9 = tid8 + offset1;
	int tid10 = tid9 + offset1;
	int tid11 = tid10 + offset1;
	int tid12 = tid11 + offset1;
	int tid13 = tid12 + offset1;
	int tid14 = tid13 + offset1;
	int tid15 = tid14 + offset1;
	int tid16 = tid15 + offset1;
	int tid17 = tid16 + offset1;
	int tid18 = tid17 + offset1;
	int tid19 = tid18 + offset1;
	int tid20 = tid19 + offset1;
	int tid21 = tid20 + offset1;
	int tid22 = tid21 + offset1;
	int tid23 = tid22 + offset1;
	int tid24 = tid23 + offset1;
	int tid25 = tid24 + offset1;
	int tid26 = tid25 + offset1;
	int tid27 = tid26 + offset1;
	int tid28 = tid27 + offset1;
	int tid29 = tid28 + offset1;
	int tid30 = tid29 + offset1;
	int tid31 = tid30 + offset1;

	float xv0 = x[tid0];   
	float xv1 = x[tid1];   
	float xv2 = x[tid2];   
	float xv3 = x[tid3];   
	float xv4 = x[tid4];   
	float xv5 = x[tid5];   
	float xv6 = x[tid6];   
	float xv7 = x[tid7];   
	float xv8 = x[tid8];   
	float xv9 = x[tid9];   
	float xv10 = x[tid10]; 
	float xv11 = x[tid11]; 
	float xv12 = x[tid12]; 
	float xv13 = x[tid13]; 
	float xv14 = x[tid14]; 
	float xv15 = x[tid15]; 
	float xv16 = x[tid16]; 
	float xv17 = x[tid17]; 
	float xv18 = x[tid18]; 
	float xv19 = x[tid19]; 
	float xv20 = x[tid20]; 
	float xv21 = x[tid21]; 
	float xv22 = x[tid22]; 
	float xv23 = x[tid23]; 
	float xv24 = x[tid24]; 
	float xv25 = x[tid25]; 
	float xv26 = x[tid26]; 
	float xv27 = x[tid27]; 
	float xv28 = x[tid28]; 
	float xv29 = x[tid29]; 
	float xv30 = x[tid30]; 
	float xv31 = x[tid31]; 
	float yv0 = y[tid0];
	float yv1 = y[tid1];
	float yv2 = y[tid2];
	float yv3 = y[tid3];
	float yv4 = y[tid4];
	float yv5 = y[tid5];
	float yv6 = y[tid6];
	float yv7 = y[tid7];
	float yv8 = y[tid8];
	float yv9 = y[tid9];
	float yv10 = y[tid10];
	float yv11 = y[tid11];
	float yv12 = y[tid12];
	float yv13 = y[tid13];
	float yv14 = y[tid14];
	float yv15 = y[tid15];
	float yv16 = y[tid16];
	float yv17 = y[tid17];
	float yv18 = y[tid18];
	float yv19 = y[tid19];
	float yv20 = y[tid20];
	float yv21 = y[tid21];
	float yv22 = y[tid22];
	float yv23 = y[tid23];
	float yv24 = y[tid24];
	float yv25 = y[tid25];
	float yv26 = y[tid26];
	float yv27 = y[tid27];
	float yv28 = y[tid28];
	float yv29 = y[tid29];
	float yv30 = y[tid30];
	float yv31 = y[tid31];


	z[tid0]   = xv0;
	z[tid1]   = xv1;
	z[tid2]   = xv2;
	z[tid3]   = xv3;
	z[tid4]   = xv4;
	z[tid5]   = xv5;
	z[tid6]   = xv6;
	z[tid7]   = xv7;
	z[tid8]   = xv8;
	z[tid9]   = xv9;
	z[tid10]  = xv10;
	z[tid11]  = xv11;
	z[tid12]  = xv12;
	z[tid13]  = xv13;
	z[tid14]  = xv14;
	z[tid15]  = xv15;
	z[tid16]  = xv16;
	z[tid17]  = xv17;
	z[tid18]  = xv18;
	z[tid19]  = xv19;
	z[tid20]  = xv20;
	z[tid21]  = xv21;
	z[tid22]  = xv22;
	z[tid23]  = xv23;
	z[tid24]  = xv24;
	z[tid25]  = xv25;
	z[tid26]  = xv26;
	z[tid27]  = xv27;
	z[tid28]  = xv28;
	z[tid29]  = xv29;
	z[tid30]  = xv30;
	z[tid31]  = xv31;

	z[tid0+N] = yv0;
	z[tid1+N] = yv1;
	z[tid2+N] = yv2;
	z[tid3+N] = yv3;
	z[tid4+N] = yv4;
	z[tid5+N] = yv5;
	z[tid6+N] = yv6;
	z[tid7+N] = yv7;
	z[tid8+N] = yv8;
	z[tid9+N] = yv9;
	z[tid10+N] = yv10;
	z[tid11+N] = yv11;
	z[tid12+N] = yv12;
	z[tid13+N] = yv13;
	z[tid14+N] = yv14;
	z[tid15+N] = yv15;
	z[tid16+N] = yv16;
	z[tid17+N] = yv17;
	z[tid18+N] = yv18;
	z[tid19+N] = yv19;
	z[tid20+N] = yv20;
	z[tid21+N] = yv21;
	z[tid22+N] = yv22;
	z[tid23+N] = yv23;
	z[tid24+N] = yv24;
	z[tid25+N] = yv25;
	z[tid26+N] = yv26;
	z[tid27+N] = yv27;
	z[tid28+N] = yv28;
	z[tid29+N] = yv29;
	z[tid30+N] = yv30;
	z[tid31+N] = yv31;




}
__global__ void concat_2d_1_same_16(
		const float *x, const float *y, float *z,
		int sizeX0, int sizeX1,
		int sizeY0, int sizeY1,
		int N)
{
	int tid00 = (blockIdx.x * blockDim.x + threadIdx.x) * 16;
	if (tid00 >= N) return;

	int tid01 = tid00 + 1;
	int tid02 = tid01 + 1;
	int tid03 = tid02 + 1;
	int tid04 = tid03 + 1;
	int tid05 = tid04 + 1;
	int tid06 = tid05 + 1;
	int tid07 = tid06 + 1;
	int tid08 = tid07 + 1;
	int tid09 = tid08 + 1;
	int tid10 = tid09 + 1;
	int tid11 = tid10 + 1;
	int tid12 = tid11 + 1;
	int tid13 = tid12 + 1;
	int tid14 = tid13 + 1;
	int tid15 = tid14 + 1;
	/*
	int offset1 = sizeX1/16;
	int tid01 = tid00 + offset1;
	int tid02 = tid01 + offset1;
	int tid03 = tid02 + offset1;
	int tid04 = tid03 + offset1;
	int tid05 = tid04 + offset1;
	int tid06 = tid05 + offset1;
	int tid07 = tid06 + offset1;
	int tid08 = tid07 + offset1;
	int tid09 = tid08 + offset1;
	int tid10 = tid09 + offset1;
	int tid11 = tid10 + offset1;
	int tid12 = tid11 + offset1;
	int tid13 = tid12 + offset1;
	int tid14 = tid13 + offset1;
	int tid15 = tid14 + offset1;
	*/

	float xv00 = x[tid00];   
	float xv01 = x[tid01];   
	float xv02 = x[tid02];   
	float xv03 = x[tid03];   
	float xv04 = x[tid04];   
	float xv05 = x[tid05];   
	float xv06 = x[tid06];   
	float xv07 = x[tid07];   
	float xv08 = x[tid08];   
	float xv09 = x[tid09];   
	float xv10 = x[tid10]; 
	float xv11 = x[tid11]; 
	float xv12 = x[tid12]; 
	float xv13 = x[tid13]; 
	float xv14 = x[tid14]; 
	float xv15 = x[tid15]; 
	float yv00 = y[tid00];
	float yv01 = y[tid01];
	float yv02 = y[tid02];
	float yv03 = y[tid03];
	float yv04 = y[tid04];
	float yv05 = y[tid05];
	float yv06 = y[tid06];
	float yv07 = y[tid07];
	float yv08 = y[tid08];
	float yv09 = y[tid09];
	float yv10 = y[tid10];
	float yv11 = y[tid11];
	float yv12 = y[tid12];
	float yv13 = y[tid13];
	float yv14 = y[tid14];
	float yv15 = y[tid15];


	z[tid00]   = xv00;
	z[tid01]   = xv01;
	z[tid02]   = xv02;
	z[tid03]   = xv03;
	z[tid04]   = xv04;
	z[tid05]   = xv05;
	z[tid06]   = xv06;
	z[tid07]   = xv07;
	z[tid08]   = xv08;
	z[tid09]   = xv09;
	z[tid10]   = xv10;
	z[tid11]   = xv11;
	z[tid12]   = xv12;
	z[tid13]   = xv13;
	z[tid14]   = xv14;
	z[tid15]   = xv15;

	z[tid00+N] = yv00;
	z[tid01+N] = yv01;
	z[tid02+N] = yv02;
	z[tid03+N] = yv03;
	z[tid04+N] = yv04;
	z[tid05+N] = yv05;
	z[tid06+N] = yv06;
	z[tid07+N] = yv07;
	z[tid08+N] = yv08;
	z[tid09+N] = yv09;
	z[tid10+N] = yv10;
	z[tid11+N] = yv11;
	z[tid12+N] = yv12;
	z[tid13+N] = yv13;
	z[tid14+N] = yv14;
	z[tid15+N] = yv15;


}


__global__ void concat_4d(
		const float *x, const float *y, float *z,
		int sizeX0, int sizeX1, int sizeX2, int sizeX3,
		int sizeY0, int sizeY1, int sizeY2, int sizeY3,
		int sizeZ0, int sizeZ1, int sizeZ2, int sizeZ3, int dim)
{
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	int N = sizeZ0 * sizeZ1 * sizeZ2 * sizeZ3;
	if (tid >= N) return;

	int z0 = tid / (sizeZ1 * sizeZ2 * sizeZ3);
	int z1 = (tid / (sizeZ2 * sizeZ3)) % sizeZ1;
	int z2 = (tid / sizeZ3) % sizeZ2;
	int z3 = tid % sizeZ3;

	const float *buf;
	int idx;
	if (dim == 0) {
		buf = (z0 < sizeX0) ? x : y;
		idx = (z0 < sizeX0) ? (tid) : (tid - sizeX0 * sizeX1 * sizeX2 * sizeX3);
	} else if (dim == 1) {
		buf = (z1 < sizeX1) ? x : y;
		int offset = z0;
		offset = (z1 < sizeX1) ? offset : offset + 1;
		int stride = (z1 < sizeX1) ? (sizeY1 * sizeY2 * sizeY3) : (sizeX1 * sizeX2 * sizeX3);
		idx = tid - offset * stride;
	} else if (dim == 2) {
		buf = (z2 < sizeX2) ? x : y;
		int offset = z0 * sizeZ1 + z1;
		offset = (z2 < sizeX2) ? offset : offset + 1;
		int stride = (z2 < sizeX2) ? (sizeY2 * sizeY3) : (sizeX2 * sizeX3);
		idx = tid - offset * stride;
	} else {
		buf = z3 < sizeX3 ? x : y;
		int offset = (z0 * sizeZ1 + z1) * sizeZ2 + z2;
		offset = (z3 < sizeX3) ? offset : offset + 1;
		int stride = (z3 < sizeX3) ? sizeY3 : sizeX3;
		idx = tid - offset * stride;
	}

	z[tid] = buf[idx];
}


EXTERNC void cuda_concat(const float *x, const float *y, float *z, int sizeX[], int sizeY[], int sizeZ[], int dim, int n)
{
	assert(n <= 4);

	int N = 1;
	for (int i = 0; i < n; i++) {
		N *= sizeZ[i];
	}

	if (n == 2) {
		if (dim == 0) {
			concat_2d_0<<< CDIV(N, 128), 128 >>>(x, y, z,
					sizeX[0], sizeX[1], sizeY[0], sizeY[1], sizeZ[0], sizeZ[1]);
		} else {
//			if (sizeX[0] == sizeY[0] && sizeX[1] == sizeY[1]) {
//				concat_2d_1_same_16<<< (sizeX[0]*sizeX[1] + 15)/16, 128 >>>(x, y, z,
//						sizeX[0], sizeX[1], sizeY[0], sizeY[1], sizeZ[0] * sizeZ[1]);
//			} else {
				concat_2d_1<<< CDIV(N, 128), 128 >>>(x, y, z,
						sizeX[0], sizeX[1], sizeY[0], sizeY[1], sizeZ[0], sizeZ[1]);
//			}
		}
	} else if (n == 3) {
		concat_4d<<< CDIV(N, 128), 128 >>>(x, y, z, sizeX[0], sizeX[1], sizeX[2], 1,
				sizeY[0], sizeY[1], sizeY[2], 1,
				sizeZ[0], sizeZ[1], sizeZ[2], 1, dim);
	} else if (n == 4) {
		concat_4d<<< CDIV(N, 128), 128 >>>(x, y, z, sizeX[0], sizeX[1], sizeX[2], sizeX[3],
				sizeY[0], sizeY[1], sizeY[2], sizeY[3],
				sizeZ[0], sizeZ[1], sizeZ[2], sizeZ[3], dim);
	} else {
		printf("Not here\n");
		exit(1);
	}
}

__global__ void split_4d(
		const float *z, float *x, float *y,
		int sizeZ0, int sizeZ1, int sizeZ2, int sizeZ3,
		int sizeX0, int sizeX1, int sizeX2, int sizeX3,
		int sizeY0, int sizeY1, int sizeY2, int sizeY3,
		int dim)
{
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	int N = sizeZ0 * sizeZ1 * sizeZ2 * sizeZ3;
	if (tid >= N) return;

	int z0 = tid / (sizeZ1 * sizeZ2 * sizeZ3);
	int z1 = (tid / (sizeZ2 * sizeZ3)) % sizeZ1;
	int z2 = (tid / sizeZ3) % sizeZ2;
	int z3 = tid % sizeZ3;

	float *buf;
	int idx;
	if (dim == 0) {
		buf = (z0 < sizeX0) ? x : y;
		idx = (z0 < sizeX0) ? (tid) : (tid - sizeX0 * sizeX1 * sizeX2 * sizeX3);
	} else if (dim == 1) {
		buf = (z1 < sizeX1) ? x : y;
		int offset = z0;
		offset = (z1 < sizeX1) ? offset : offset + 1;
		int stride = (z1 < sizeX1) ? (sizeY1 * sizeY2 * sizeY3) : (sizeX1 * sizeX2 * sizeX3);
		idx = tid - offset * stride;
	} else if (dim == 2) {
		buf = (z2 < sizeX2) ? x : y;
		int offset = z0 * sizeZ1 + z1;
		offset = (z2 < sizeX2) ? offset : offset + 1;
		int stride = (z2 < sizeX2) ? (sizeY2 * sizeY3) : (sizeX2 * sizeX3);
		idx = tid - offset * stride;
	} else {
		buf = z3 < sizeX3 ? x : y;
		int offset = (z0 * sizeZ1 + z1) * sizeZ2 + z2;
		offset = (z3 < sizeX3) ? offset : offset + 1;
		int stride = (z3 < sizeX3) ? sizeY3 : sizeX3;
		idx = tid - offset * stride;
	}

	buf[idx] = z[tid];
}

EXTERNC void cuda_split(const float *z, float *x, float *y, int sizeZ[], int sizeX[], int sizeY[], int dim, int n) 
{
	assert(n <= 4);

	int N = 1;
	for (int i = 0; i < n; i++) {
		N *= sizeZ[i];
	}

	if (n == 2) {
		split_4d<<< CDIV(N, 128), 128 >>>(z, x, y, sizeZ[0], sizeZ[1], 1, 1,
				sizeX[0], sizeX[1], 1, 1,
				sizeY[0], sizeY[1], 1, 1, dim);
	} else if (n == 3) {
		split_4d<<< CDIV(N, 128), 128 >>>(z, x, y, sizeZ[0], sizeZ[1], sizeZ[2], 1,
				sizeX[0], sizeX[1], sizeX[2], 1,
				sizeY[0], sizeY[1], sizeY[2], 1, dim);
	} else if (n == 4) {
		split_4d<<< CDIV(N, 128), 128 >>>(z, x, y, sizeZ[0], sizeZ[1], sizeZ[2], sizeZ[3],
				sizeX[0], sizeX[1], sizeX[2], sizeX[3],
				sizeY[0], sizeY[1], sizeY[2], sizeY[3], dim);
	} else {
		printf("Not here\n");
		exit(1);
	}

	chkCUDA(cudaGetLastError());
}


__global__ void expand_4d(
		const float *x, float *y,
		int sizeX0, int sizeX1, int sizeX2, int sizeX3,
		int sizeY0, int sizeY1, int sizeY2, int sizeY3,
		int dim, int size)
{
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	int N = sizeY0 * sizeY1 * sizeY2;
	if (tid >= N) return;

	int y0 = (tid / (sizeY1 * sizeY2 * sizeY3));
	int y1 = (tid / (sizeY2 * sizeY3)) % sizeY1;
	int y2 = (tid / sizeY3) % sizeY2;
	int y3 = tid % sizeY3;

	int x0 = y0;
	int x1 = y1;
	int x2 = y2;
	int x3 = y3;

	if (dim == 0) x0 = 0;
	else if (dim == 1) x1 = 0;
	else if (dim == 2) x2 = 0;
	else x3 = 0;

	int idx = ((x0 * sizeX1 + x1) * sizeX2 + x2) * sizeX3 + x3;

	y[tid] = x[idx];
}

EXTERNC void cuda_expand(const float *x, float *y, int sizeY[], int dim, int size, int n)
{
	assert(n <= 4);
	assert(dim <= n);
	int sizeX[n];
	int N = 1;
	for (int i = 0; i < n; i++) {
		sizeX[i] = sizeY[i];
		N *= sizeY[i];
	}
	sizeX[dim] = 1;

	if (n == 2) {
		expand_4d<<< CDIV(N, 128), 128 >>>(x, y,
				sizeX[0], sizeX[1], 1, 1,
				sizeY[0], sizeY[1], 1, 1,
				dim, size);
	} else if (n == 3) {
		expand_4d<<< CDIV(N, 128), 128 >>>(x, y,
				sizeX[0], sizeX[1], sizeX[2], 1,
				sizeY[0], sizeY[1], sizeY[2], 1,
				dim, size);
	} else if (n == 4) {
		expand_4d<<< CDIV(N, 128), 128 >>>(x, y,
				sizeX[0], sizeX[1], sizeX[2], sizeX[3],
				sizeY[0], sizeY[1], sizeY[2], sizeY[3],
				dim, size);
	} else {
		printf("Fuck\n");
		exit(1);
	}

	chkCUDA(cudaGetLastError());
}


__global__ void embedding_forward(const float *w, const int *indicies, float *y, int N, int embed_dim)
{
	int wrow = indicies[blockIdx.x] * embed_dim;
	int yrow = blockIdx.x * embed_dim;

	for (int i = threadIdx.x; i < embed_dim; i += blockDim.x) {
		y[yrow + i] = w[wrow + i];
	}
}

EXTERNC void cuda_embedding_forward(const float *w, const int *indicies, float *y, int N, int embed_dim)
{
	// w: N x E
	// y: B x N x E
	// ind: B x N
	embedding_forward<<< N, 128 >>>(w, indicies, y, N, embed_dim);
	chkCUDA(cudaGetLastError());
}

__global__ void embedding_backward(const float *dy, const int *indicies, float *dw, int N, int embed_dim) 
{
	int wrow = indicies[blockIdx.x] * embed_dim;
	int yrow = blockIdx.x * embed_dim;

	for (int i = threadIdx.x; i < embed_dim; i += blockDim.x) {
		atomicAdd(&dw[wrow + i], dy[yrow + i]);
	}
}

EXTERNC void cuda_embedding_backward(const float *dy, const int *indicies, float *dw, int N, int embed_dim) 
{
	embedding_backward<<< N, 128 >>>(dy, indicies, dw, N, embed_dim);
	chkCUDA(cudaGetLastError());
}


__global__ void bias_forward(const float *bias, float *y, int N, int M)
{
	int m = blockDim.x * blockIdx.x + threadIdx.x;
	int n = blockDim.y * blockIdx.y + threadIdx.y;
	if (n >= N || m >= M) return;


	y[n * M + m] += bias[m];
}

EXTERNC void cuda_bias_forward(const float *bias, float *y, int N, int M)
{
	// y: N x M
	// bias: M
	dim3 nThreads(16, 16);
	dim3 nBlocks(CDIV(M, 16), CDIV(N, 16));
	bias_forward<<< nBlocks, nThreads >>>(bias, y, N, M);
	chkCUDA(cudaGetLastError());
}

__global__ void masked_copy(float *y, const int *mask, float val, int N)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= N) return;

	if (mask[i] != 0)
		y[i] = val;
}

EXTERNC void cuda_masked_copy(float *y, const int *mask, float val, int N)
{
	masked_copy<<< CDIV(N, 128), 128 >>>(y, mask, val, N);
	chkCUDA(cudaGetLastError());
}


__global__ void slice_4d(
		const float *x, float *y,
		int sizeX0, int sizeX1, int sizeX2, int sizeX3,
		int sizeY0, int sizeY1, int sizeY2, int sizeY3,
		int idx, int dim)
{
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	int N = sizeY0 * sizeY1 * sizeY2;
	if (tid >= N) return;

	int y0 = (tid / (sizeY1 * sizeY2 * sizeY3));
	int y1 = (tid / (sizeY2 * sizeY3)) % sizeY1;
	int y2 = (tid / sizeY3) % sizeY2;
	int y3 = tid % sizeY3;

	int x0 = y0;
	int x1 = y1;
	int x2 = y2;
	int x3 = y3;

	if (dim == 0) x0 = idx;
	else if (dim == 1) x1 = idx;
	else if (dim == 2) x2 = idx;
	else x3 = idx;

	int xidx = ((x0 * sizeX1 + x1) * sizeX2 + x2) * sizeX3 + x3;
	y[tid] = x[xidx];
}

EXTERNC void cuda_slice(const float *x, float *y, const int sizeX[], int n, int idx, int dim)
{
	assert(n <= 4);
	assert(dim <= n);
	int sizeY[n];
	int N = 1;
	for (int i = 0; i < n; i++) {
		sizeY[i] = sizeX[i];
		N *= sizeY[i];
	}
	sizeY[dim] = 1;

	if (n == 2) {
		slice_4d<<< CDIV(N, 128), 128 >>>(x, y,
				sizeX[0], sizeX[1], 1, 1,
				sizeY[0], sizeY[1], 1, 1,
				idx, dim);
	} else if (n == 3) {
		slice_4d<<< CDIV(N, 128), 128 >>>(x, y,
				sizeX[0], sizeX[1], sizeX[2], 1,
				sizeY[0], sizeY[1], sizeY[2], 1,
				idx, dim);
	} else if (n == 4) {
		slice_4d<<< CDIV(N, 128), 128 >>>(x, y,
				sizeX[0], sizeX[1], sizeX[2], sizeX[3],
				sizeY[0], sizeY[1], sizeY[2], sizeY[3],
				idx, dim);
	} else {
		printf("Fuck\n");
		exit(1);
	}

	chkCUDA(cudaGetLastError());
}

__global__ void mask(const int *x, int *y, int m, int N)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= N) return;
	y[i] = (x[i] == m) ? 1 : 0;
}

EXTERNC void cuda_mask(const int *x, int *y, int m, int N)
{
	mask<<< CDIV(N, 128), 128 >>>(x, y, m, N);
	chkCUDA(cudaGetLastError());
}

__global__ void scale(float *m, size_t N, float min, float max)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= N) return;
	m[i] = (max - min) * m[i] + min;
}

EXTERNC void cuda_scale(float *m, size_t N, float min, float max)
{
	scale<<< CDIV(N, 128), 128 >>>(m, N, min, max);
	chkCUDA(cudaGetLastError());
}
