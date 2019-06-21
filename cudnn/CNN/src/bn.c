#include <math.h>
#include <time.h>
#include <string.h>

#include <cuda_runtime_api.h>
#include <cuda.h>
#include <cudnn.h>

#include "layer.h"
#include "params.h"
#include "utils.h"
#include "memory.h"
#include "execute.h"

void init_bn_layer(
    bn_layer *l, const char *name,
    int batch_size, int channel, int height, int width, int nth)
{
  ////////////////////////////////////////////////////////////////
  // 1. Initialize Parameters
  ////////////////////////////////////////////////////////////////
  strcpy(l->name, name);

  l->batch_size = batch_size;
  l->channel = channel;
  l->height = height;
  l->width = width;

  l->mode = CUDNN_BATCHNORM_SPATIAL;

  l->eaf = 1.0 / (1.0 + nth);
  l->eps = CUDNN_BN_MIN_EPSILON;

  l->input = NULL;
  l->d_input = NULL;

  l->output = NULL;
  l->d_output = NULL;

  l->bias = NULL;
  l->d_bias = NULL;

  l->scale = NULL;
  l->d_scale = NULL;

  l->running_mean = NULL;
  l->running_var = NULL;

  l->save_mean = NULL;
  l->save_var = NULL;

  clear_time_bn_layer(l);

  ////////////////////////////////////////////////////////////////
  // 2. Create Tensors
  ////////////////////////////////////////////////////////////////
  create_buffer_data(
      &l->input, CUDNN_DATA_FLOAT, 4,
      l->batch_size, l->channel, l->height, l->width);

  create_buffer_data(
      &l->d_input, CUDNN_DATA_FLOAT, 4,
      l->batch_size, l->channel, l->height, l->width);

  create_buffer_data(
      &l->output, CUDNN_DATA_FLOAT, 4,
      l->batch_size, l->channel, l->height, l->width);

  create_buffer_data_gradient(
      &l->d_output, CUDNN_DATA_FLOAT, 4,
      l->batch_size, l->channel, l->height, l->width);

  ////////////////////////////////////////////////////////////////
  // 3. Create BN Params
  ////////////////////////////////////////////////////////////////
  create_buffer_bn_param(
      &l->scale, CUDNN_DATA_FLOAT, l->mode, 4,
      l->batch_size, l->channel, l->height, l->width);

  create_buffer_bn_param_gradient(
      &l->d_scale, CUDNN_DATA_FLOAT, l->mode, 4,
      l->batch_size, l->channel, l->height, l->width);

  create_buffer_bn_param(
      &l->bias, CUDNN_DATA_FLOAT, l->mode, 4,
      l->batch_size, l->channel, l->height, l->width);

  create_buffer_bn_param_gradient(
      &l->d_bias, CUDNN_DATA_FLOAT, l->mode, 4,
      l->batch_size, l->channel, l->height, l->width);

  create_buffer_bn_param(
      &l->running_mean, CUDNN_DATA_FLOAT, l->mode, 4,
      l->batch_size, l->channel, l->height, l->width);

  create_buffer_bn_param(
      &l->running_var, CUDNN_DATA_FLOAT, l->mode, 4,
      l->batch_size, l->channel, l->height, l->width);

  create_buffer_bn_param(
      &l->save_mean, CUDNN_DATA_FLOAT, l->mode, 4,
      l->batch_size, l->channel, l->height, l->width);

  create_buffer_bn_param(
      &l->save_var, CUDNN_DATA_FLOAT, l->mode, 4,
      l->batch_size, l->channel, l->height, l->width);
}

void train_fwd_bn_layer(bn_layer *l)
{
  START_CNN_TIMER(fwd_t);
  execute_bn_fwd(
      l->mode, l->eaf, l->eps, l->input, l->output, l->scale, l->bias,
      l->running_mean, l->running_var, l->save_mean, l->save_var);
  STOP_CNN_TIMER(fwd_t);
}

void train_bwd_bn_layer(bn_layer *l)
{
  START_CNN_TIMER(bwd_t);
  execute_bn_bwd(
      l->mode, l->eaf, l->eps, l->input, l->d_output, l->d_input,
      l->scale, l->d_scale, l->d_bias, l->save_mean, l->save_var);
  STOP_CNN_TIMER(bwd_t);

  START_CNN_TIMER(bwd_update_t);
  execute_apply_gradient(params.learning_rate, l->d_scale, l->scale);
  execute_apply_gradient(params.learning_rate, l->d_bias, l->bias);
  STOP_CNN_TIMER(bwd_update_t);
}

// l->mode == CUDNN_BATCHNORM_SPATIAL
size_t param_size_bn(bn_layer *l)
{
  int count = l->channel * 2;
  return data_type_size(l->scale) * count;
}

// l->mode == CUDNN_BATCHNORM_SPATIAL
int set_bn_param(bn_layer *l, float *param)
{
  write_buffer(l->scale, param, true);
  write_buffer(l->bias, param + l->channel, true);
  return l->channel * 2;
}

// l->mode == CUDNN_BATCHNORM_SPATIAL
int get_bn_param(bn_layer *l, float *param)
{
  read_buffer(param, l->scale, true);
  read_buffer(param + l->channel, l->bias, true);
  return l->channel * 2;
}

void print_time_bn_layer(bn_layer *l)
{
  printf("%s, %.3f, %.3f, %.3f, %.3f\n",
      l->name, l->fwd_t, l->bwd_t, 0.0f, l->bwd_update_t);
}

void clear_time_bn_layer(bn_layer *l)
{
  l->fwd_t = 0.0;
  l->bwd_t = 0.0;
  l->bwd_update_t = 0.0;
}
