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

void init_branch_layer(
    branch_layer *l, int batch_size, int fan_out,
    int channel, int height, int width)
{
  ////////////////////////////////////////////////////////////////
  // 1. Initialize Parameters
  ////////////////////////////////////////////////////////////////
  l->batch_size = batch_size;
  l->channel = channel;
  l->height = height;
  l->width = width;
  l->batch = batch;

  l->fan_out = fan_out;

  l->input = NULL;
  l->d_input = NULL;

  for (int i = 0; i < l->fan_out; i++) {
    l->d_output[i] = NULL;
  }

  clear_time_branch_layer(l);

  ////////////////////////////////////////////////////////////////
  // 2. Set OpTensor Descriptor
  ////////////////////////////////////////////////////////////////
  chkCUDNN(cudnnCreateOpTensorDescriptor(&l->op_desc));

  chkCUDNN(cudnnSetOpTensorDescriptor(
        l->op_desc, CUDNN_OP_TENSOR_ADD, CUDNN_DATA_FLOAT, CUDNN_NOT_PROPAGATE_NAN));

  ////////////////////////////////////////////////////////////////
  // 3. Create Tensors
  ////////////////////////////////////////////////////////////////
  create_buffer[DATA](
      &l->input, 4, CUDNN_DATA_FLOAT, l->batch_size,
      l->channel, l->height, l->width);

  create_buffer[DATA_GRADIENT](
      &l->d_input, 4, CUDNN_DATA_FLOAT, l->batch_size,
      l->channel, l->height, l->width);

  for (int i = 0; i < l->fan_out; i++) {
    create_buffer[DATA_GRADIENT](
        &l->d_output[i], 4, CUDNN_DATA_FLOAT, l->batch_size,
        l->channel, l->height, l->width);
  }
}

void train_fwd_branch_layer(branch_layer *l)
{
  START_CNN_TIMER(fwd_t);

  STOP_CNN_TIMER(fwd_t);
}

void train_bwd_branch_layer(branch_layer *l)
{
  START_CNN_TIMER(bwd_t);
  execute_branch(l->op_desc, l->fan_out, l->d_output, l->d_input);
  STOP_CNN_TIMER(bwd_t);
}

void print_time_branch_layer(branch_layer *l, char *name)
{
  printf("%s, %.3f, %.3f, %.3f, %.3f\n",
      name, l->fwd_t, l->bwd_t, 0.0f, 0.0f);
}

void clear_time_branch_layer(branch_layer *l)
{
  l->fwd_t = 0.0;
  l->bwd_t = 0.0;
}
