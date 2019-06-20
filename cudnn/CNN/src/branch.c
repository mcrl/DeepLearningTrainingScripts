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

void init_branch_layer(
    branch_layer *l, const char *name,
    int batch_size, int fan_out, int channel, int height, int width)
{
  ////////////////////////////////////////////////////////////////
  // 1. Initialize Parameters
  ////////////////////////////////////////////////////////////////
  strcpy(l->name, name);

  l->batch_size = batch_size;
  l->channel = channel;
  l->height = height;
  l->width = width;

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
        l->op_desc, CUDNN_OP_TENSOR_ADD,
        CUDNN_DATA_FLOAT, CUDNN_NOT_PROPAGATE_NAN));

  ////////////////////////////////////////////////////////////////
  // 3. Create Tensors
  ////////////////////////////////////////////////////////////////
  create_buffer_data(
      &l->input, CUDNN_DATA_FLOAT, 4,
      l->batch_size, l->channel, l->height, l->width);

  create_buffer_data_gradient(
      &l->d_input, CUDNN_DATA_FLOAT, 4,
      l->batch_size, l->channel, l->height, l->width);

  for (int i = 0; i < l->fan_out; i++) {
    create_buffer_data_gradient(
        &l->d_output[i], CUDNN_DATA_FLOAT, 4,
        l->batch_size, l->channel, l->height, l->width);
  }
}

void train_fwd_branch_layer(branch_layer *l)
{
}

void train_bwd_branch_layer(branch_layer *l)
{
  START_CNN_TIMER(bwd_t);
  execute_branch_bwd(l->op_desc, l->fan_out, l->d_output, l->d_input);
  STOP_CNN_TIMER(bwd_t);
}

void print_time_branch_layer(branch_layer *l)
{
  printf("%s, %.3f, %.3f, %.3f, %.3f\n",
      l->name, 0.0f, l->bwd_t, 0.0f, 0.0f);
}

void clear_time_branch_layer(branch_layer *l)
{
  l->bwd_t = 0.0;
}
