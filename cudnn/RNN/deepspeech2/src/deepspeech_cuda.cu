#include <string.h>
#include <stdio.h>
#include <math.h>

#include <cuda.h>
#include <cuda_runtime_api.h>
#include <cuda_profiler_api.h>
#include <cublas_v2.h>

#include "deepspeech_cuda.h"
#include "utils.h"
#include "params.h"

#include "kernels.h"

extern cublasHandle_t cublas;

#define CDIV(a,b) (((a) + (b) - 1) / (b))

void deepspeech_cuda_transpose(float *in, float *out, int N, int H, int W)
{
  dim3 size_block(8, 8, 8);
  dim3 size_grid(CDIV(N, size_block.x), CDIV(H, size_block.y),
    CDIV(W, size_block.z));

  transpose_3d<<<size_grid, size_block>>>(in, out, N, H, W);
}

void deepspeech_cuda_transpose_inverse(float *in, float *out,
  int N, int H, int W)
{
  dim3 size_block(8, 8, 8);
  dim3 size_grid(CDIV(N, size_block.x), CDIV(H, size_block.y),
    CDIV(W, size_block.z));

  transpose_inverse_3d<<<size_grid, size_block>>>(in, out, N, H, W);
}

void deepspeech_cuda_pack_padded_seq(float *in, float *out,
  int input_size, int batch_size, int seq_length, int *cnt_seqs)
{
  float *offset_in = in;
  float *offset_out = out;

  chkCUDA(cudaDeviceSynchronize());
  for (int t = 0; t < seq_length; t++) {
    chkCUDA(cudaMemcpy(offset_out, offset_in,
      sizeof(float) * input_size * cnt_seqs[t], cudaMemcpyDeviceToDevice));
    offset_in += batch_size * input_size;
    offset_out += cnt_seqs[t] * input_size;
  }
}

void deepspeech_cuda_pad_packed_seq(float *in, float *out,
  int input_size, int batch_size, int seq_length, int *cnt_seqs)
{
  float *offset_in = in;
  float *offset_out = out;

  for (int t = 0; t < seq_length; t++) {
    chkCUDA(cudaMemcpy(offset_out, offset_in,
      sizeof(float) * input_size * cnt_seqs[t], cudaMemcpyDeviceToDevice));
    offset_in += cnt_seqs[t] * input_size;
    offset_out += batch_size * input_size;
  }
}

void deepspeech_cuda_sum_padded_seq(float *in, float *out,
  int input_size, int batch_size, int seq_length)
{
  assert(input_size % 2 == 0);

  dim3 size_block(256);
  dim3 size_grid(((input_size / 2) + size_block.x - 1) / size_block.x);

  sum_padded_seq_1d<<<size_grid, size_block>>>(in, out,
    input_size, seq_length * batch_size);
}

void deepspeech_cuda_expand_sum_padded_seq(float *in, float *out,
  int input_size, int batch_size, int seq_length)
{
  assert(input_size % 2 == 0);

  dim3 size_block(256);
  dim3 size_grid(((input_size / 2) + size_block.x - 1) / size_block.x);

  expand_sum_padded_seq_1d<<<size_grid, size_block>>>(in, out,
    input_size, seq_length * batch_size);
}

void deepspeech_cuda_apply_grad(float *x, float *dx, float **buf_dx,
  float lr, float clip, int N, int C, int H, int W)
{
  assert(lr != 0);

  const size_t bytes = sizeof(float) * N * C * H * W;
  const size_t num_elems = N * C * H * W;

  const float clip_inverse = 1 / clip;
  const float weight_decay = params.weight_decay;
  const float momentum = params.momentum;
  const float one = 1.0;
  const float learning_rate = -lr;

  chkCUBLAS(cublasSscal(cublas, num_elems, &clip_inverse, dx, 1));
  chkCUBLAS(cublasSaxpy(cublas, num_elems, &weight_decay, x, 1, dx, 1));

  if ((*buf_dx) == NULL) {
    chkCUDA(cudaMalloc((void **)buf_dx, bytes));
    chkCUDA(cudaMemset(*buf_dx, 0, bytes));
  }

  chkCUBLAS(cublasSscal(cublas, num_elems, &momentum, *buf_dx, 1));
  chkCUBLAS(cublasSaxpy(cublas, num_elems, &one, dx, 1, *buf_dx, 1));
  chkCUBLAS(cublasSaxpy(cublas, num_elems, &momentum, *buf_dx, 1, dx, 1));
  chkCUBLAS(cublasSaxpy(cublas, num_elems, &learning_rate, dx, 1, x, 1));
}

#define SIZE_BLOCK_X 512
#define SIZE_GRID_X 40

float deepspeech_cuda_sum_square(float *t, int N, int C, int H, int W)
{
  static float *out = NULL;
  float result[SIZE_GRID_X];
  float _result = 0;
  int len = N * C * H * W;

  dim3 size_block(SIZE_BLOCK_X);
  dim3 size_grid(SIZE_GRID_X);

  if (out == NULL) {
    chkCUDA(cudaMalloc((void **)&out, sizeof(float) * size_grid.x));
  }

  START_STOPWATCH {
    snrm2<<<size_grid, size_block, sizeof(float) * size_block.x>>>(t, out, len,
      CDIV(len, size_block.x * size_grid.x));
    chkCUDA(cudaMemcpy(&result, out, sizeof(float) * size_grid.x,
      cudaMemcpyDeviceToHost));

    for (int i = 0; i < size_grid.x; i++) {
      _result += result[i];  
    }
  } STOP_STOPWATCH("cuda sum square");

  return sqrt(_result);
}

