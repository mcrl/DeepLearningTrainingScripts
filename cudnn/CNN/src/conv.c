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

void init_conv_layer(
    conv_layer *l, const char *name,
    int batch_size, int filter_height, int filter_width,
    int pad_height, int pad_width, int stride_height, int stride_width,
    int input_channel, int output_channel, int input_height, int input_width)
{
  size_t ws_fwd_size;
  size_t ws_bwd_data_size;
  size_t ws_bwd_filter_size;

  ////////////////////////////////////////////////////////////////
  // 1. Initialize Parameters
  ////////////////////////////////////////////////////////////////
  strcpy(l->name, name);

  l->filter_height = filter_height;
  l->filter_width = filter_width;
  l->pad_height = pad_height;
  l->pad_width = pad_width;
  l->stride_height = stride_height;
  l->stride_width = stride_width;

  l->batch_size = batch_size;
  l->input_channel = input_channel;
  l->input_height = input_height;
  l->input_width = input_width;
  l->output_channel = output_channel;
  l->output_height = (input_height + pad_height * 2 - filter_height) / stride_height + 1;
  l->output_width = (input_width + pad_width * 2 - filter_width) / stride_width + 1;

  l->input = NULL;
  l->d_input = NULL;

  l->output = NULL;
  l->d_output = NULL;

  l->filter = NULL;
  l->d_filter = NULL;

  l->ws_fwd = NULL;
  l->ws_bwd_data = NULL;
  l->ws_bwd_filter = NULL;

  clear_time_conv_layer(l);

  ////////////////////////////////////////////////////////////////
  // 2. Set Convolution Descriptor
  ////////////////////////////////////////////////////////////////
  chkCUDNN(cudnnCreateConvolutionDescriptor(&l->conv_desc));

  chkCUDNN(cudnnSetConvolution2dDescriptor(
        l->conv_desc, l->pad_height, l->pad_width,
        l->stride_height, l->stride_width, 1, 1,
        CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT));

  ////////////////////////////////////////////////////////////////
  // 3. Create Tensors & Filters
  ////////////////////////////////////////////////////////////////
  create_buffer_data(
      &l->input, CUDNN_DATA_FLOAT, 4,
      l->batch_size, l->input_channel, l->input_height, l->input_width);

  create_buffer_data_gradient(
      &l->d_input, CUDNN_DATA_FLOAT, 4,
      l->batch_size, l->input_channel, l->input_height, l->input_width);

  create_buffer_data(
      &l->output, CUDNN_DATA_FLOAT, 4,
      l->batch_size, l->output_channel, l->output_height, l->output_width);

  create_buffer_data_gradient(
      &l->d_output, CUDNN_DATA_FLOAT, 4,
      l->batch_size, l->output_channel, l->output_height, l->output_width);

  create_buffer_weight(
      &l->filter, CUDNN_DATA_FLOAT, 4,
      l->output_channel, l->input_channel,
      l->filter_height, l->filter_width);
  
  create_buffer_weight_gradient(
      &l->d_filter, CUDNN_DATA_FLOAT, 4,
      l->output_channel, l->input_channel,
      l->filter_height, l->filter_width);

  ////////////////////////////////////////////////////////////////
  // 4. Get Convolution Algorithm
  ////////////////////////////////////////////////////////////////
  execute_get_conv_fwd_algo(
      l->conv_desc, l->input, l->filter, l->output, &l->fwd_algo);

  execute_get_conv_bwd_data_algo(
      l->conv_desc, l->filter, l->d_output, l->d_input, &l->bwd_data_algo);

  execute_get_conv_bwd_filter_algo(
      l->conv_desc, l->input, l->d_output, l->d_filter, &l->bwd_filter_algo);

  ////////////////////////////////////////////////////////////////
  // 5. Get Work Space Size in Bytes
  ////////////////////////////////////////////////////////////////
  execute_get_conv_fwd_ws_size(
      l->conv_desc, l->fwd_algo,
      l->input, l->filter, l->output, &ws_fwd_size);

  execute_get_conv_bwd_data_ws_size(
      l->conv_desc, l->bwd_data_algo,
      l->filter, l->d_output, l->d_input, &ws_bwd_data_size);

  execute_get_conv_bwd_filter_ws_size(
      l->conv_desc, l->bwd_filter_algo,
      l->input, l->d_output, l->d_filter, &ws_bwd_filter_size);

  ////////////////////////////////////////////////////////////////
  // 6. Create Work Spaces
  ////////////////////////////////////////////////////////////////
  create_buffer_work_space(&l->ws_fwd, ws_fwd_size);

  create_buffer_work_space(&l->ws_bwd_data, ws_bwd_data_size);

  create_buffer_work_space(&l->ws_bwd_filter, ws_bwd_filter_size);
}

void train_fwd_conv_layer(conv_layer *l)
{
  START_CNN_TIMER(fwd_t);
  execute_conv_fwd(
      l->conv_desc, l->fwd_algo, l->input, l->filter, l->output, l->ws_fwd);
  STOP_CNN_TIMER(fwd_t);
}

void train_bwd_conv_layer(conv_layer *l)
{
  START_CNN_TIMER(bwd_data_t);
  execute_conv_bwd_data(
      l->conv_desc, l->bwd_data_algo, l->filter, l->d_output, l->d_input, l->ws_bwd_data);
  STOP_CNN_TIMER(bwd_data_t);

  START_CNN_TIMER(bwd_filter_t);
  execute_conv_bwd_filter(
      l->conv_desc, l->bwd_filter_algo, l->input, l->d_output, l->d_filter, l->ws_bwd_filter);
  STOP_CNN_TIMER(bwd_filter_t);

  START_CNN_TIMER(bwd_update_t);
  execute_apply_gradient(params.learning_rate, l->d_filter, l->filter);
  STOP_CNN_TIMER(bwd_update_t);
}

size_t param_size_conv(conv_layer *l)
{
  int count = l->filter_height * l->filter_width * l->input_channel * l->output_channel;
  return data_type_size(l->filter) * count;
}

int set_conv_filter(conv_layer *l, float *filter)
{
  write_buffer(l->filter, filter, true);
  return l->filter_height * l->filter_width * l->input_channel * l->output_channel;
}

int get_conv_filter(conv_layer *l, float *filter)
{
  read_buffer(filter, l->filter, true);
  return l->filter_height * l->filter_width * l->input_channel * l->output_channel;
}

void print_time_conv_layer(conv_layer *l)
{
  printf("%s, %.3f, %.3f, %.3f, %.3f\n",
      l->name, l->fwd_t, l->bwd_data_t, l->bwd_filter_t, l->bwd_update_t);
}

void clear_time_conv_layer(conv_layer *l)
{
  l->fwd_t = 0.0;
  l->bwd_data_t = 0.0;
  l->bwd_filter_t = 0.0;
  l->bwd_update_t = 0.0;
}
