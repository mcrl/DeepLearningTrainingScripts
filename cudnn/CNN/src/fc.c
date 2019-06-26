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

void init_fc_layer(
    fc_layer *l, const char *name, int batch_size, int in, int out)
{
  size_t ws_fwd_size;
  size_t ws_bwd_data_size;
  size_t ws_bwd_filter_size;

  ////////////////////////////////////////////////////////////////
  // 1. Initialize Parameters
  ////////////////////////////////////////////////////////////////
  strcpy(l->name, name);

  l->batch_size = batch_size;
  l->in = in;
  l->out = out;

  l->input = NULL;
  l->d_input = NULL;

  l->output = NULL;
  l->d_output = NULL;

  l->weight = NULL;
  l->d_weight = NULL;

  l->ws_fwd = NULL;
  l->ws_bwd_data = NULL;
  l->ws_bwd_filter = NULL;

  clear_time_fc_layer(l);

  ////////////////////////////////////////////////////////////////
  // 2. Set Convolution Descriptor
  ////////////////////////////////////////////////////////////////
  chkCUDNN(cudnnCreateConvolutionDescriptor(&l->conv_desc));

  chkCUDNN(cudnnSetConvolution2dDescriptor(
        l->conv_desc, 0, 0, 1, 1, 1, 1,
        CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT));

  ////////////////////////////////////////////////////////////////
  // 3. Create Tensors
  ////////////////////////////////////////////////////////////////
  create_buffer_data(
      &l->input, CUDNN_DATA_FLOAT, 4, l->batch_size, l->in, 1, 1);

  create_buffer_data_gradient(
      &l->d_input, CUDNN_DATA_FLOAT, 4, l->batch_size, l->in, 1, 1);

  create_buffer_data(
      &l->output, CUDNN_DATA_FLOAT, 4, l->batch_size, l->out, 1, 1);

  create_buffer_data_gradient(
      &l->d_output, CUDNN_DATA_FLOAT, 4, l->batch_size, l->out, 1, 1);

  ////////////////////////////////////////////////////////////////
  // 4. Create Weights
  ////////////////////////////////////////////////////////////////
  create_buffer_weight(
      &l->weight, CUDNN_DATA_FLOAT, 4, l->out, l->in, 1, 1);

  create_buffer_weight_gradient(
      &l->d_weight, CUDNN_DATA_FLOAT, 4, l->out, l->in, 1, 1);

  ////////////////////////////////////////////////////////////////
  // 5. Get Convolution Algorithm
  ////////////////////////////////////////////////////////////////
  execute_get_conv_fwd_algo(
      l->conv_desc, l->input, l->weight, l->output, &l->fwd_algo);

  execute_get_conv_bwd_data_algo(
      l->conv_desc, l->weight, l->d_output, l->d_input, &l->bwd_data_algo);
  //l->bwd_data_algo = CUDNN_CONVOLUTION_BWD_DATA_ALGO_1; // FIXME

  execute_get_conv_bwd_filter_algo(
      l->conv_desc, l->input, l->d_output, l->d_weight, &l->bwd_filter_algo);
  //l->bwd_filter_algo = CUDNN_CONVOLUTION_BWD_FILTER_ALGO_1; // FIXME

  ////////////////////////////////////////////////////////////////
  // 6. Get Work Space Size in Bytes
  ////////////////////////////////////////////////////////////////
  execute_get_conv_fwd_ws_size(
      l->conv_desc, l->fwd_algo,
      l->input, l->weight, l->output, &ws_fwd_size);

  execute_get_conv_bwd_data_ws_size(
      l->conv_desc, l->bwd_data_algo,
      l->weight, l->d_output, l->d_input, &ws_bwd_data_size);

  execute_get_conv_bwd_filter_ws_size(
      l->conv_desc, l->bwd_filter_algo,
      l->input, l->d_output, l->d_weight, &ws_bwd_filter_size);

  ////////////////////////////////////////////////////////////////
  // 7. Create Work Spaces
  ////////////////////////////////////////////////////////////////
  create_buffer_work_space(&l->ws_fwd, ws_fwd_size);

  create_buffer_work_space(&l->ws_bwd_data, ws_bwd_data_size);

  create_buffer_work_space(&l->ws_bwd_filter, ws_bwd_filter_size);
}

void train_fwd_fc_layer(fc_layer *l)
{
  START_CNN_TIMER(fwd_t);
  execute_conv_fwd(
      l->conv_desc, l->fwd_algo, l->input, l->weight, l->output, l->ws_fwd);
  STOP_CNN_TIMER(fwd_t);
}

void train_bwd_fc_layer(fc_layer *l)
{
  START_CNN_TIMER(bwd_data_t);
  execute_conv_bwd_data(
      l->conv_desc, l->bwd_data_algo, l->weight, l->d_output, l->d_input, l->ws_bwd_data);
  STOP_CNN_TIMER(bwd_data_t);

  START_CNN_TIMER(bwd_weight_t);
  execute_conv_bwd_filter(
      l->conv_desc, l->bwd_filter_algo, l->input, l->d_output, l->d_weight, l->ws_bwd_filter);
  STOP_CNN_TIMER(bwd_weight_t);

  START_CNN_TIMER(bwd_update_t);
  execute_apply_gradient(params.learning_rate, l->d_weight, l->weight);
  STOP_CNN_TIMER(bwd_update_t);
}

size_t param_size_fc(fc_layer *l)
{
  int count = l->in * l->out;
  return data_type_size(l->weight) * count;
}

int set_fc_weight(fc_layer *l, float *weight)
{
  write_buffer(l->weight, weight, true);
  return l->in * l->out;
}

int get_fc_weight(fc_layer *l, float *weight)
{
  read_buffer(weight, l->weight, true);
  return l->in * l->out;
}

void print_time_fc_layer(fc_layer *l)
{
  printf("%s, %.3f, %.3f, %.3f, %.3f\n",
      l->name, l->fwd_t, l->bwd_data_t, l->bwd_weight_t, l->bwd_update_t);
}

void clear_time_fc_layer(fc_layer *l)
{
  l->fwd_t = 0.0;
  l->bwd_data_t = 0.0;
  l->bwd_weight_t = 0.0;
  l->bwd_update_t = 0.0;
}
