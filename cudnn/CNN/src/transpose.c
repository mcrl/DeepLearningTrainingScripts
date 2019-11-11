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

void init_transpose_layer(
    transpose_layer *l, const char *name,
    int batch_size, int channel, int height, int width,
    int axisX, int axisY)
{
  ////////////////////////////////////////////////////////////////
  // 1. Initialize Parameters
  ////////////////////////////////////////////////////////////////
  strcpy(l->name, name);

  l->batch_size = batch_size;
  l->channel = channel;
  l->height = height;
  l->width = width;

  l->axisX = axisX;
  l->axisY = axisY;

  l->input = NULL;
  l->d_input = NULL;

  l->output = NULL;
  l->d_output = NULL;

  clear_time_transpose_layer(l);

  ////////////////////////////////////////////////////////////////
  // 2. Create Tensors
  ////////////////////////////////////////////////////////////////
  create_buffer_data(
      &l->input, CUDNN_DATA_FLOAT, 4,
      l->batch_size, l->channel, l->height, l->width);

  create_buffer_data_gradient(
      &l->d_input, CUDNN_DATA_FLOAT, 4,
      l->batch_size, l->channel, l->height, l->width);

  create_buffer_data(
      &l->output, CUDNN_DATA_FLOAT, 4,
      l->batch_size, l->channel, l->height, l->width);

  create_buffer_data_gradient(
      &l->d_output, CUDNN_DATA_FLOAT, 4,
      l->batch_size, l->channel, l->height, l->width);
}

void train_fwd_transpose_layer(transpose_layer *l)
{
  START_CNN_TIMER(fwd_t);
  execute_transpose(l->input, l->output, l->axisX, l->axisY);
  STOP_CNN_TIMER(fwd_t);
}

void train_bwd_transpose_layer(transpose_layer *l)
{
  START_CNN_TIMER(bwd_t);
  execute_transpose(l->d_output, l->d_input, l->axisX, l->axisY);
  STOP_CNN_TIMER(bwd_t);
}

void print_time_transpose_layer(transpose_layer *l)
{
  printf("%s, %.3f, %.3f, %.3f, %.3f\n",
      l->name, l->fwd_t, l->bwd_t, 0.0f, 0.0f);
}

void clear_time_transpose_layer(transpose_layer *l)
{
  l->fwd_t = 0.0;
  l->bwd_t = 0.0;
}
