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

void init_pool_layer(
    pool_layer *l, int batch_size, int filter_height, int filter_width, 
    int pad_height, int pad_width, int stride_height, int stride_width,
    int channel, int input_height, int input_width, pool_type type)
{
  ////////////////////////////////////////////////////////////////
  // 1. Initialize Parameters
  ////////////////////////////////////////////////////////////////
  l->filter_height = filter_height;
  l->filter_width = filter_width;
  l->pad_height = pad_height;
  l->pad_width = pad_width;
  l->stride_height = stride_height;
  l->stride_width = stride_width;

  l->batch_size = batch_size;
  l->channel = channel;
  l->input_height = input_height;
  l->input_width = input_width;
  l->output_height = (input_height + pad_height * 2 - filter_height) / stride_height + 1;
  l->output_width = (input_width + pad_width * 2 - filter_width) / stride_width + 1;

  l->type = type;

  l->input = NULL;
  l->d_input = NULL;

  l->output = NULL;
  l->d_output = NULL;

  clear_time_pool_layer(l);

  ////////////////////////////////////////////////////////////////
  // 2. Set Pooling Descriptor
  ////////////////////////////////////////////////////////////////
  chkCUDNN(cudnnCreatePoolingDescriptor(&l->pooling_desc));

  // l->type == MAX_T
  chkCUDNN(cudnnSetPooling2dDescriptor(
        l->pool_desc, CUDNN_POOLING_MAX_DETERMINISTIC,
        CUDNN_NOT_PROPAGATE_NAN, filter_height, filter_width,
        pad_height, pad_width, stride_height, stride_width));

  ////////////////////////////////////////////////////////////////
  // 3. Create Tensors
  ////////////////////////////////////////////////////////////////
  create_buffer[DATA](
      &l->input, 4, CUDNN_DATA_FLOAT, l->batch_size,
      l->channel, l->input_height, l->input_width);

  create_buffer[DATA_GRADIENT](
      &l->d_input, 4, CUDNN_DATA_FLOAT, l->batch_size,
      l->channel, l->input_height, l->input_width);

  create_buffer[DATA](
      &l->output, 4, CUDNN_DATA_FLOAT, l->batch_size,
      l->channel, l->output_height, l->output_width);

  create_buffer[DATA_GRADIENT](
      &l->d_output, 4, CUDNN_DATA_FLOAT, l->batch_size,
      l->channel, l->output_height, l->output_width);
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
  execute_pool_fwd(
      l->pool_desc, l->output, l->d_output, l->input, l->d_input);
  STOP_CNN_TIMER(bwd_t);
}

void print_time_pool_layer(pool_layer *l, char *name)
{
  printf("%s, %.3f, %.3f, %.3f, %.3f\n",
      name, l->fwd_t, l->bwd_t, 0.0f, 0.0f);
}

void clear_time_pool_layer(pool_layer *l)
{
  l->fwd_t = 0.0;
  l->bwd_t = 0.0;
}
