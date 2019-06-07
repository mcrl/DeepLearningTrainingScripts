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

void init_bias_layer(
    bias_layer *l, cudnnHandle_t cudnn,
    int batch, int channel, int height, int width)
{
  l->cudnn = cudnn;
  l->width = width;
  l->height = height;
  l->channel = channel;

  l->output_desc = NULL;
  l->d_output_desc = NULL;

  l->fwd_t = 0;
  l->bwd_t = 0;
  l->bwd_update_t = 0;

  chkCUDNN(cudnnCreateTensorDescriptor(&l->bias_desc));
  chkCUDNN(cudnnCreateTensorDescriptor(&l->d_bias_desc));
  chkCUDNN(cudnnCreateTensorDescriptor(&l->output_desc));
  chkCUDNN(cudnnCreateTensorDescriptor(&l->d_output_desc));

  MALLOC_TENSOR_FLOAT(&l->bias, 1, channel, 1, 1);
  MALLOC_TENSOR_FLOAT(&l->d_bias, 1, channel, 1, 1);

  chkCUDNN(cudnnSetTensor4dDescriptor(
        l->output_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
        batch, channel, height, width));

  chkCUDNN(cudnnSetTensor4dDescriptor(
        l->d_output_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
        batch, channel, height, width));

  chkCUDNN(cudnnSetTensor4dDescriptor(
        l->bias_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
        1, channel, 1, 1));

  chkCUDNN(cudnnSetTensor4dDescriptor(
        l->d_bias_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
        1, channel, 1, 1));
}

void train_fwd_bias_layer(bias_layer *l)
{
  START_CNN_TIMER(fwd_t);

  chkCUDNN(cudnnAddTensor(
        l->cudnn, &one, l->bias_desc, l->bias,
        &one, l->output_desc, l->output));

  STOP_CNN_TIMER(fwd_t);
}

void train_bwd_bias_layer(bias_layer *l)
{
  START_CNN_TIMER(bwd_t);

  chkCUDNN(cudnnConvolutionBackwardBias(
        l->cudnn, &one, l->d_output_desc, l->d_output,
        &one, l->d_bias_desc, l->d_bias));

  STOP_CNN_TIMER(bwd_t);

  START_CNN_TIMER(bwd_update_t);

  cublas_apply_grad(l->bias, l->d_bias, params.learning_rate,  l->channel);

  STOP_CNN_TIMER(bwd_update_t);
}

int set_bias(bias_layer l, float *bias)
{
  size_t s = PSIZE_BIAS(l);
  chkCUDA(cudaMemcpy(l.bias, bias, s, cudaMemcpyHostToDevice));
  return s / sizeof(float);
}

int get_bias(bias_layer l, float *bias)
{
  size_t s = PSIZE_BIAS(l);
  chkCUDA(cudaMemcpy(bias, l.bias, s, cudaMemcpyDeviceToHost));
  return s / sizeof(float);
}

void print_time_bias_layer(bias_layer *l, char *name)
{
  printf("%s, %.3f, %.3f, %.3f, %.3f\n",
      name, l->fwd_t, 0.0f, l->bwd_t, l->bwd_update_t);
}

void clear_time_bias_layer(bias_layer *l)
{
  l->fwd_t = 0.0;
  l->bwd_t = 0.0;
  l->bwd_update_t = 0.0;
}
