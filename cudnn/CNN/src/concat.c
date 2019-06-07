#include <math.h>
#include <time.h>

#include <cuda_runtime_api.h>
#include <cuda.h>
#include <cudnn.h>

#include "cnn.h"
#include "cnn_cuda.h"
#include "layer.h"
#include "params.h"
#include "utils.h"

static float one = 1.0f;
static float zero = 0.0f;

void init_concat_layer(
    concat_layer *l, cudnnHandle_t cudnn,
    int batch, int in_cnt, int* channel_in, int height, int width)
{
  l->cudnn = cudnn;
  l->width = width;
  l->height = height;
  l->batch = batch;
  l->in_cnt = in_cnt;

  l->fwd_t = 0;
  l->bwd_t = 0;

  int c = 0;
  for (int i = 0; i < in_cnt; i++) {
    l->channel_in[i] = channel_in[i];
    c += channel_in[i];

    l->input[i] = NULL;

    chkCUDNN(cudnnCreateTensorDescriptor(&l->input_desc[i]));
    chkCUDNN(cudnnCreateTensorDescriptor(&l->d_input_desc[i]));

    MALLOC_TENSOR_FLOAT(&l->d_input[i], batch, l->channel_in[i], height, width);

    chkCUDNN(cudnnSetTensor4dDescriptor(
          l->input_desc[i], CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
          batch, channel_in[i], height, width));
    chkCUDNN(cudnnSetTensor4dDescriptor(
          l->d_input_desc[i], CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
          batch, channel_in[i], height, width));
  }

  l->channel = c;

  chkCUDNN(cudnnCreateTensorDescriptor(&l->output_desc));
  chkCUDNN(cudnnCreateTensorDescriptor(&l->d_output_desc));

  MALLOC_TENSOR_FLOAT(&l->output, batch, l->channel, height, width);
  MALLOC_TENSOR_FLOAT(&l->d_output, batch, l->channel, height, width);

  chkCUDNN(cudnnSetTensor4dDescriptor(
        l->output_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
        batch, l->channel, height, width));

  chkCUDNN(cudnnSetTensor4dDescriptor(
        l->d_output_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
        batch, l->channel, height, width));
}

void train_fwd_concat_layer(concat_layer *l)
{
  START_CNN_TIMER(fwd_t);

  cuda_concat(
      l->batch, l->in_cnt, l->channel_in,
      l->height, l->width, 1, l->input, l->output);

  STOP_CNN_TIMER(fwd_t);
}

void train_bwd_concat_layer(concat_layer *l)
{
  START_CNN_TIMER(bwd_t);

  cuda_concat(
      l->batch, l->in_cnt, l->channel_in,
      l->height, l->width, 1, l->d_input, l->d_output);

  STOP_CNN_TIMER(bwd_t);
}

void print_time_concat_layer(concat_layer *l, char *name)
{
  printf("%s, %.3f, %.3f, %.3f, %.3f\n",
      name, l->fwd_t, l->bwd_t, 0.0f, 0.0f);
}

void clear_time_concat_layer(concat_layer *l)
{
  l->fwd_t = 0.0;
  l->bwd_t = 0.0;
}
