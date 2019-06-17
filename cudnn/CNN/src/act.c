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

void init_act_layer(
    act_layer *l, int batch_size, int channel, int height, int width, act_type type)
{
  ////////////////////////////////////////////////////////////////
  // 1. Initialize Parameters
  ////////////////////////////////////////////////////////////////
  l->batch_size = batch_size;
  l->channel = channel;
  l->height = height;
  l->width = width;

  l->type = type;

  l->input = NULL;
  l->d_input = NULL;

  l->output = NULL;
  l->d_output = NULL;

  clear_time_act_layer(l);

  ////////////////////////////////////////////////////////////////
  // 2. Set Activation Descriptor
  ////////////////////////////////////////////////////////////////
  chkCUDNN(cudnnCreateActivationDescriptor(&l->act_desc));

  // l->type == RELU_T
  chkCUDNN(cudnnSetActivationDescriptor(
        l->act_desc, CUDNN_ACTIVATION_RELU, CUDNN_NOT_PROPAGATE_NAN, 20.0));

  ////////////////////////////////////////////////////////////////
  // 3. Create Tensors
  ////////////////////////////////////////////////////////////////
  create_buffer[DATA](
      &l->input, 4, CUDNN_DATA_FLOAT, l->batch_size,
      l->channel, l->height, l->width);

  create_buffer[DATA_GRADIENT](
      &l->d_input, 4, CUDNN_DATA_FLOAT, l->batch_size,
      l->channel, l->height, l->width);

  create_buffer[DATA](
      &l->output, 4, CUDNN_DATA_FLOAT, l->batch_size,
      l->channel, l->height, l->width);

  create_buffer[DATA_GRADIENT](
      &l->d_output, 4, CUDNN_DATA_FLOAT, l->batch_size,
      l->channel, l->height, l->width);
}

void train_fwd_act_layer(act_layer *l)
{
  START_CNN_TIMER(fwd_t);
  execute_act_fwd(l->act_desc, l->input, l->output);
  STOP_CNN_TIMER(fwd_t);
}

void train_bwd_act_layer(act_layer *l)
{
  START_CNN_TIMER(bwd_t);
  execute_act_bwd(l->act_desc, l->output, l->d_output, l->input, l->d_input);
  STOP_CNN_TIMER(bwd_t);
}

void print_time_act_layer(act_layer *l, char *name)
{
  printf("%s, %.3f, %.3f, %.3f, %.3f\n",
      name, l->fwd_t, l->bwd_t, 0.0f, 0.0f);
}

void clear_time_act_layer(act_layer *l)
{
  l->fwd_t = 0.0;
  l->bwd_t = 0.0;
}
