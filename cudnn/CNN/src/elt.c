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

void init_elt_layer(
    elt_layer *l, int batch_size, int channel, int height, int width, elt_type type);
{
  ////////////////////////////////////////////////////////////////
  // 1. Initialize Parameters
  ////////////////////////////////////////////////////////////////
  l->batch_size = batch_size;
  l->channel = channel;
  l->height = height;
  l->width = width;

  l->type = type;

  l->input[0] = NULL;
  l->input[1] = NULL;

  l->output = NULL;
  l->d_output = NULL;

  clear_time_elt_layer(l);

  ////////////////////////////////////////////////////////////////
  // 2. Set OpTensor Descriptor
  ////////////////////////////////////////////////////////////////
  chkCUDNN(cudnnCreateOpTensorDescriptor(&l->op_desc));

  // l->type == ADD_T
  chkCUDNN(cudnnSetOpTensorDescriptor(
        l->op_desc, CUDNN_OP_TENSOR_ADD, CUDNN_DATA_FLOAT, CUDNN_NOT_PROPAGATE_NAN));

  ////////////////////////////////////////////////////////////////
  // 3. Create Tensors
  ////////////////////////////////////////////////////////////////
  create_buffer[DATA](
      &l->input[0], 4, CUDNN_DATA_FLOAT, l->batch_size,
      l->channel, l->height, l->width);

  create_buffer[DATA](
      &l->input[1], 4, CUDNN_DATA_FLOAT, l->batch_size,
      l->channel, l->height, l->width);

  create_buffer[DATA](
      &l->output, 4, CUDNN_DATA_FLOAT, l->batch_size,
      l->channel, l->height, l->width);

  create_buffer[DATA_GRADIENT](
      &l->d_output, 4, CUDNN_DATA_FLOAT, l->batch_size,
      l->channel, l->height, l->width);
}

void train_fwd_elt_layer(elt_layer *l)
{
  START_CNN_TIMER(fwd_t);
  execute_elt(l->op_desc, l->input[0], l->input[1], l->output);
  STOP_CNN_TIMER(fwd_t);
}

void train_bwd_elt_layer(elt_layer *l)
{
  START_CNN_TIMER(bwd_t);

  STOP_CNN_TIMER(bwd_t);
}

void print_time_elt_layer(elt_layer *l, char *name)
{
  printf("%s, %.3f, %.3f, %.3f, %.3f\n",
      name, l->fwd_t, l->bwd_t, 0.0f, 0.0f);
}

void clear_time_elt_layer(elt_layer *l)
{
  l->fwd_t = 0.0;
  l->bwd_t = 0.0;
}
