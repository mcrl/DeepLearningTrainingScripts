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
#include "memory.h"
#include "execute.h"

void init_bias_layer(
    bias_layer *l, int batch_size, int channel, int height, int width)
{
  ////////////////////////////////////////////////////////////////
  // 1. Initialize Parameters
  ////////////////////////////////////////////////////////////////
  l->batch_size = batch_size;
  l->channel = channel;
  l->height = height;
  l->width = width;

  l->output = NULL;
  l->d_output = NULL;

  l->bias = NULL;
  l->d_bias = NULL;

  clear_time_bias_layer(l);

  ////////////////////////////////////////////////////////////////
  // 2. Create Tensors
  ////////////////////////////////////////////////////////////////
  create_buffer[DATA](
      &l->output, 4, CUDNN_DATA_FLOAT, l->batch_size,
      l->channel, l->height, l->width);

  create_buffer[DATA_GRADIENT](
      &l->d_output, 4, CUDNN_DATA_FLOAT, l->batch_size,
      l->channel, l->height, l->width);

  create_buffer[WEIGHT](
      &l->bias, 4, CUDNN_DATA_FLOAT, 1, l->channel, 1, 1);

  create_buffer[WEIGHT_GRADIENT](
      &l->d_bias, 4, CUDNN_DATA_FLOAT, 1, l->channel, 1, 1);
}

void train_fwd_bias_layer(bias_layer *l)
{
  START_CNN_TIMER(fwd_t);
  execute_bias_fwd(l->bias, l->output);
  STOP_CNN_TIMER(fwd_t);
}

void train_bwd_bias_layer(bias_layer *l)
{
  START_CNN_TIMER(bwd_t);
  execute_bias_bwd(l->d_output, l->d_bias);
  STOP_CNN_TIMER(bwd_t);

  START_CNN_TIMER(bwd_update_t);
  execute_apply_gradient(params.learning_rate, l->d_bias, l->bias);
  STOP_CNN_TIMER(bwd_update_t);
}

int set_bias(bias_layer *l, float *bias)
{
  write_buffer(l->bias, bias, true);
  return l->channel;
}

int get_bias(bias_layer l, float *bias)
{
  read_buffer(bias, l->bias, true);
  return l->channel;
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
