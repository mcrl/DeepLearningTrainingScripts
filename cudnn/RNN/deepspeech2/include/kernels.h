#ifndef _KERNELS_H_
#define _KERNELS_H_

__global__ void sum_padded_seq_1d(float *in, float *out, int len, int num_rows);
__global__ void sum_padded_seq_2d(float *in, float *out, int len);

__global__ void expand_sum_padded_seq_1d(float *in, float *out, int len, int num_rows);

__global__ void transpose_3d(float *in, float *out, int N, int H, int W);
__global__ void transpose_inverse_3d(float *in, float *out, int N, int H, int W);

__global__ void snrm2(float *in, float *out, int len, int elem_per_th);

#endif

