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

static float one = 1.0f;
static float zero = 0.0f;

void init_elt_layer(
    elt_layer *l, cudnnHandle_t cudnn,
    int batch, int channel, int height, int width, ELT_TYPE type)
{
  l->cudnn = cudnn;
  l->width = width;
  l->height = height;
  l->channel = channel;
  l->batch = batch;

  l->input1 = NULL;
  l->input2 = NULL;
  l->d_output = NULL;

  l->fwd_t = 0;
  l->bwd_t = 0;

  chkCUDNN(cudnnCreateTensorDescriptor(&l->input1_desc));
  chkCUDNN(cudnnCreateTensorDescriptor(&l->input2_desc));
  chkCUDNN(cudnnCreateTensorDescriptor(&l->output_desc));
  chkCUDNN(cudnnCreateTensorDescriptor(&l->d_output_desc));

  chkCUDNN(cudnnCreateOpTensorDescriptor(&l->op_desc));

  MALLOC_TENSOR_FLOAT(&l->output, batch, channel, height, width);

  chkCUDNN(cudnnSetTensor4dDescriptor(
        l->input1_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
        batch, l->channel, height, width));

  chkCUDNN(cudnnSetTensor4dDescriptor(
        l->input2_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
        batch, l->channel, height, width));

  chkCUDNN(cudnnSetTensor4dDescriptor(
        l->output_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
        batch, l->channel, height, width));

  chkCUDNN(cudnnSetTensor4dDescriptor(
        l->d_output_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
        batch, l->channel, height, width));

  chkCUDNN(cudnnSetOpTensorDescriptor(
        l->op_desc, CUDNN_OP_TENSOR_ADD, CUDNN_DATA_FLOAT, CUDNN_NOT_PROPAGATE_NAN));
}

void train_fwd_elt_layer(elt_layer *l)
{
  START_CNN_TIMER(fwd_t);

  chkCUDNN(cudnnOpTensor(l->cudnn, l->op_desc, 
    &one, l->input1_desc, l->input1,
    &one, l->input2_desc, l->input2,
    &zero, l->output_desc, l->output));

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
