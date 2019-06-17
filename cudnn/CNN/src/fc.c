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

void init_fc_layer(
    fc_layer *l, int batch_size, int in, int out)
{
  size_t ws_fwd_size;
  size_t ws_bwd_data_size;
  size_t ws_bwd_filter_size;

  ////////////////////////////////////////////////////////////////
  // 1. Initialize Parameters
  ////////////////////////////////////////////////////////////////
  l->batch_size = batch_size;
  l->in = in;
  l->out = out;

  l->input = NULL;
  l->d_input = NULL;

  l->output = NULL;
  l->d_output = NULL;

  l->filter = NULL;
  l->d_filter = NULL;

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
  // 3. Create Tensors & Filters
  ////////////////////////////////////////////////////////////////
  create_buffer[DATA](
      &l->input, 4, CUDNN_DATA_FLOAT, l->batch_size, l->in, 1, 1);

  create_buffer[DATA_GRADIENT](
      &l->d_input, 4, CUDNN_DATA_FLOAT, l->batch_size, l->in, 1, 1);

  create_buffer[DATA](
      &l->output, 4, CUDNN_DATA_FLOAT, l->batch_size, l->out, 1, 1);

  create_buffer[DATA_GRADIENT](
      &l->d_output, 4, CUDNN_DATA_FLOAT, l->batch_size, l->out, 1, 1);

  create_buffer[WEIGHT](
      &l->filter, 4, CUDNN_DATA_FLOAT, l->out, l->in, 1, 1);

  create_buffer[WEIGHT_GRADIENT](
      &l->d_filter, 4, CUDNN_DATA_FLOAT, l->out, l->in, 1, 1);

  ////////////////////////////////////////////////////////////////
  // 4. Get Convolution Algorithm
  ////////////////////////////////////////////////////////////////
  execute_get_conv_fwd_algo(
      l->conv_desc, l->input, l->filter, l->output, &l->fwd_aglo);

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
  create_buffer[WORK_SPACE](&l->ws_fwd, 1, ws_fwd_size);

  create_buffer[WORK_SPACE](&l->ws_bwd_data, 1, ws_bwd_data_size);

  create_buffer[WORK_SPACE](&l->ws_bwd_filter, 1, ws_bwd_filter_size);
}

void train_fwd_fc_layer(fc_layer *l)
{
  START_CNN_TIMER(fwd_t);
  execute_conv_fwd(
      l->conv_desc, l->fwd_algo, l->input, l->filter, l->output, l->ws_fwd);
  STOP_CNN_TIMER(fwd_t);
}

void train_bwd_fc_layer(fc_layer *l)
{
  START_CNN_TIMER(bwd_data_t);
  execute_conv_bwd_data(
      l->conv_desc, l->bwd_data_algo, l->filter, l->d_output, l->d_input, l->ws_bwd_data);
  STOP_CNN_TIMER(bwd_data_t);

  START_CNN_TIMER(bwd_weight_t);
  execute_conv_bwd_filter(
      l->conv_desc, l->bwd_filter_algo, l->input, l->d_output, l->d_filter, l->ws_bwd_filter);
  STOP_CNN_TIMER(bwd_weight_t);

  START_CNN_TIMER(bwd_update_t);
  execute_apply_gradient(params.learning_rate, l->d_filter, l->filter);
  STOP_CNN_TIMER(bwd_update_t);
}

int set_fc_weight(fc_layer *l, float *weight)
{
  write_buffer(l->filter, weight, true);
  return l->in * l->out;
}

int get_fc_weight(fc_layer *l, float *weight)
{
  read_buffer(weight, l->filter, true);
  return l->in * l->out;
}

void print_time_fc_layer(fc_layer *l, char *name)
{
  printf("%s, %.3f, %.3f, %.3f, %.3f\n",
      name, l->fwd_t, l->bwd_data_t, l->bwd_weight_t, l->bwd_update_t);
}

void clear_time_fc_layer(fc_layer *l)
{
  l->fwd_t = 0.0;
  l->bwd_data_t = 0.0;
  l->bwd_weight_t = 0.0;
  l->bwd_update_t = 0.0;
}
