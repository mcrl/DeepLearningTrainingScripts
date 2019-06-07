#include <string.h>
#include <stdio.h>
#include <math.h>

#include <cuda.h>
#include <cuda_runtime_api.h>
#include <cuda_profiler_api.h>
#include <cublas_v2.h>

#include "cnn_cuda.h"
#include "utils.h"
#include "params.h"

#include "kernels.h"

extern cublasHandle_t cublas;

//static float one = 1.0f;
//static float zero = 0.0f;

void cublas_apply_grad(float *x, float *dx, float lr, int num_elems)
{
  const float learning_rate = -lr;

  chkCUBLAS(cublasSaxpy(cublas, num_elems, &learning_rate, dx, 1, x, 1));
}

void cuda_set_label(int batch_size, int class_cnt, int *label_in, float *label)
{
  dim3 size_block(256);
  dim3 size_grid((batch_size * class_cnt + 255) / 256);
  set_label<<<size_grid, size_block>>>(batch_size, class_cnt, label_in, label);
  cudaDeviceSynchronize();
}

void cuda_concat(
    int batch_size, int in_cnt, int *channel_in, int height, int width, int fwd, float **in, float *out)
{
  int csum = 0;
  for (int i = 0; i < in_cnt; i++) {
    csum += channel_in[i];
  }

  dim3 size_block(256);
  dim3 size_grid((batch_size * csum * height * width + 255) / 256);

  if (in_cnt == 2) {
    concat2<<<size_grid, size_block>>>(
        batch_size, channel_in[0], channel_in[1], height, width, fwd, in[0], in[1], out);
    cudaDeviceSynchronize();
  }
}
