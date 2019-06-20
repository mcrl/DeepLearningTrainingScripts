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

static const cudnnPoolingMode_t pool_mode[] = {
  CUDNN_POOLING_MAX_DETERMINISTIC,
  CUDNN_POOLING_AVERAGE_COUNT_EXCLUDE_PADDING
};

void init_pool_layer(
    pool_layer *l, const char *name,
    int batch_size, int window_height, int window_width, 
    int pad_height, int pad_width, int stride_height, int stride_width,
    int channel, int input_height, int input_width, pool_type type)
{
  ////////////////////////////////////////////////////////////////
  // 1. Initialize Parameters
  ////////////////////////////////////////////////////////////////
  strcpy(l->name, name);

  l->window_height = window_height;
  l->window_width = window_width;
  l->pad_height = pad_height;
  l->pad_width = pad_width;
  l->stride_height = stride_height;
  l->stride_width = stride_width;

  l->batch_size = batch_size;
  l->channel = channel;
  l->input_height = input_height;
  l->input_width = input_width;
  l->output_height = (input_height + pad_height * 2 - window_height) / stride_height + 1;
  l->output_width = (input_width + pad_width * 2 - window_width) / stride_width + 1;

  l->type = type;

  l->input = NULL;
  l->d_input = NULL;

  l->output = NULL;
  l->d_output = NULL;

  clear_time_pool_layer(l);

  ////////////////////////////////////////////////////////////////
  // 2. Set Pooling Descriptor
  ////////////////////////////////////////////////////////////////
  chkCUDNN(cudnnCreatePoolingDescriptor(&l->pool_desc));

  chkCUDNN(cudnnSetPooling2dDescriptor(
        l->pool_desc, pool_mode[l->type],
        CUDNN_NOT_PROPAGATE_NAN, window_height, window_width,
        pad_height, pad_width, stride_height, stride_width));

  ////////////////////////////////////////////////////////////////
  // 3. Create Tensors
  ////////////////////////////////////////////////////////////////
  create_buffer_data(
      &l->input, CUDNN_DATA_FLOAT, 4,
      l->batch_size, l->channel, l->input_height, l->input_width);

  create_buffer_data_gradient(
      &l->d_input, CUDNN_DATA_FLOAT, 4,
      l->batch_size, l->channel, l->input_height, l->input_width);

  create_buffer_data(
      &l->output, CUDNN_DATA_FLOAT, 4,
      l->batch_size, l->channel, l->output_height, l->output_width);

  create_buffer_data_gradient(
      &l->d_output, CUDNN_DATA_FLOAT, 4,
      l->batch_size, l->channel, l->output_height, l->output_width);
}

void train_fwd_pool_layer(pool_layer *l)
{
  START_CNN_TIMER(fwd_t);
  execute_pool_fwd(l->pool_desc, l->input, l->output);
  STOP_CNN_TIMER(fwd_t);
}

void train_bwd_pool_layer(pool_layer *l)
{
  START_CNN_TIMER(bwd_t);
  execute_pool_bwd(
      l->pool_desc, l->output, l->d_output, l->input, l->d_input);
  STOP_CNN_TIMER(bwd_t);
}

void print_time_pool_layer(pool_layer *l)
{
  printf("%s, %.3f, %.3f, %.3f, %.3f\n",
      l->name, l->fwd_t, l->bwd_t, 0.0f, 0.0f);
}

void clear_time_pool_layer(pool_layer *l)
{
  l->fwd_t = 0.0;
  l->bwd_t = 0.0;
}
