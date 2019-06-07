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

void init_bn_layer(
    bn_layer *l, cudnnHandle_t cudnn,
    int batch, int channel, int height, int width)
{
  l->cudnn = cudnn;
  l->channel = channel;
  l->bn_eafactor = 0.1;
  l->bn_epsilon = 1e-05;

  l->is_training = true;

  l->input = NULL;
  l->d_output = NULL;

  l->fwd_t = 0;
  l->bwd_t = 0;
  l->bwd_update_t = 0;

  chkCUDNN(cudnnCreateTensorDescriptor(&l->input_desc));
  chkCUDNN(cudnnCreateTensorDescriptor(&l->input_desc));
  chkCUDNN(cudnnCreateTensorDescriptor(&l->d_input_desc));
  chkCUDNN(cudnnCreateTensorDescriptor(&l->output_desc));
  chkCUDNN(cudnnCreateTensorDescriptor(&l->d_output_desc));
  chkCUDNN(cudnnCreateTensorDescriptor(&l->bn_desc));
  chkCUDNN(cudnnCreateTensorDescriptor(&l->d_bn_desc));

  MALLOC_TENSOR_FLOAT(&l->bn_scale, 1, channel, 1, 1);
  MALLOC_TENSOR_FLOAT(&l->bn_bias, 1, channel, 1, 1);
  MALLOC_TENSOR_FLOAT(&l->d_bn_scale, 1, channel, 1, 1);
  MALLOC_TENSOR_FLOAT(&l->d_bn_bias, 1, channel, 1, 1);
  MALLOC_TENSOR_FLOAT(&l->bn_result_running_mean, 1, channel, 1, 1);
  MALLOC_TENSOR_FLOAT(&l->bn_result_running_var, 1, channel, 1, 1);

  float *tmp = (float *)malloc(sizeof(float) * channel);

  for (int i = 0; i < channel; i++) {
    tmp[i] = 1.0f;
  }

  chkCUDA(cudaMemcpy(
        l->bn_result_running_var, tmp, sizeof(float) * channel, cudaMemcpyHostToDevice)); 
  free(tmp);

  MALLOC_TENSOR_FLOAT(&l->bn_result_save_mean, 1, channel, 1, 1);
  MALLOC_TENSOR_FLOAT(&l->bn_result_save_var, 1, channel, 1, 1);
  MALLOC_TENSOR_FLOAT(&l->output, batch, channel, height, width);
  MALLOC_TENSOR_FLOAT(&l->d_input, batch, channel, height, width);

  chkCUDNN(cudnnSetTensor4dDescriptor(
        l->input_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
        batch, channel, height, width));

  chkCUDNN(cudnnSetTensor4dDescriptor(
        l->d_input_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
        batch, channel, height, width));

  chkCUDNN(cudnnSetTensor4dDescriptor(
        l->output_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
        batch, channel, height, width));

  chkCUDNN(cudnnSetTensor4dDescriptor(
        l->d_output_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
        batch, channel, height, width));

  chkCUDNN(cudnnDeriveBNTensorDescriptor(
        l->bn_desc, l->output_desc, CUDNN_BATCHNORM_SPATIAL));

  chkCUDNN(cudnnDeriveBNTensorDescriptor(
        l->d_bn_desc, l->d_output_desc, CUDNN_BATCHNORM_SPATIAL));
}

void train_fwd_bn_layer(bn_layer *l)
{
  START_CNN_TIMER(fwd_t);

  if (l->is_training) {
    chkCUDNN(cudnnBatchNormalizationForwardTraining(
          l->cudnn, CUDNN_BATCHNORM_SPATIAL, &one, &zero,
          l->input_desc, l->input,
          l->output_desc, l->output, l->bn_desc,
          l->bn_scale, l->bn_bias, l->bn_eafactor,
          l->bn_result_running_mean, l->bn_result_running_var,
          l->bn_epsilon, l->bn_result_save_mean, l->bn_result_save_var));
  }
  else {
    chkCUDNN(cudnnBatchNormalizationForwardInference(
          l->cudnn, CUDNN_BATCHNORM_SPATIAL, &one, &zero,
          l->input_desc, l->input,
          l->output_desc, l->output, l->bn_desc,
          l->bn_scale, l->bn_bias,
          l->bn_result_running_mean, l->bn_result_running_var,
          l->bn_epsilon));
  }

  STOP_CNN_TIMER(fwd_t);
}

void train_bwd_bn_layer(bn_layer *l)
{
  START_CNN_TIMER(bwd_t);

  chkCUDNN(cudnnBatchNormalizationBackward(
        l->cudnn, CUDNN_BATCHNORM_SPATIAL, &one, &zero, &one, &zero,
        l->input_desc, l->input,
        l->d_output_desc, l->d_output,
        l->d_input_desc, l->d_input,
        l->d_bn_desc, l->bn_scale, l->d_bn_scale, l->d_bn_bias,
        l->bn_epsilon, l->bn_result_save_mean, l->bn_result_save_var));

  STOP_CNN_TIMER(bwd_t);

  START_CNN_TIMER(bwd_update_t);

  cublas_apply_grad(l->bn_scale, l->d_bn_scale, 0,  l->channel);
  cublas_apply_grad(l->bn_bias, l->d_bn_bias, 0,  l->channel);

  STOP_CNN_TIMER(bwd_update_t);
}

int set_bn_vars(bn_layer l, float *bn)
{
  size_t s = PSIZE_BN(l) / 2;
  chkCUDA(cudaMemcpy(l.bn_scale, bn, s, cudaMemcpyHostToDevice));
  chkCUDA(cudaMemcpy(l.bn_bias, bn + l.channel, s, cudaMemcpyHostToDevice));
  return s * 2 / sizeof(float);
}

int get_bn_vars(bn_layer l, float *bn)
{
  size_t s = PSIZE_BN(l) / 6;
  chkCUDA(cudaMemcpy(bn, l.bn_scale, s, cudaMemcpyDeviceToHost));
  chkCUDA(cudaMemcpy(bn + l.channel, l.bn_bias, s, cudaMemcpyDeviceToHost));
  return s * 2 / sizeof(float);
}

void print_time_bn_layer(bn_layer *l, char *name)
{
  printf("%s, %.3f, %.3f, %.3f, %.3f\n",
      name, l->fwd_t, l->bwd_t, 0.0f, l->bwd_update_t);
}

void clear_time_bn_layer(bn_layer *l)
{
  l->fwd_t = 0.0;
  l->bwd_t = 0.0;
  l->bwd_update_t = 0.0;
}
