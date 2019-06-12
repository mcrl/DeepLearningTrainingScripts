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

void init_branch_layer(
    branch_layer *l, cudnnHandle_t cudnn,
    int batch, int out_cnt, int channel, int height, int width)
{
  l->cudnn = cudnn;
  l->width = width;
  l->height = height;
  l->channel = channel;
  l->batch = batch;
  l->fwd_t = l->bwd_t = 0;
  l->out_cnt = out_cnt;

  l->input = NULL;

  for (int i = 0; i < out_cnt; i++) {
    l->d_output[i] = NULL;
  }

  chkCUDNN(cudnnCreateTensorDescriptor(&l->input_desc));
  chkCUDNN(cudnnCreateTensorDescriptor(&l->d_input_desc));
  chkCUDNN(cudnnCreateTensorDescriptor(&l->d_output_desc));
  chkCUDNN(cudnnCreateOpTensorDescriptor(&l->op_desc));

  MALLOC_TENSOR_FLOAT(&l->d_input, batch, channel, height, width);

  chkCUDNN(cudnnSetTensor4dDescriptor(
        l->input_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
        batch, l->channel, height, width));

  chkCUDNN(cudnnSetTensor4dDescriptor(
        l->d_input_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
        batch, l->channel, height, width));

  chkCUDNN(cudnnSetTensor4dDescriptor(
        l->d_output_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
        batch, l->channel, height, width));

  chkCUDNN(cudnnSetOpTensorDescriptor(
        l->op_desc, CUDNN_OP_TENSOR_ADD, CUDNN_DATA_FLOAT, CUDNN_NOT_PROPAGATE_NAN));
}

void train_fwd_branch_layer(branch_layer *l)
{
  START_CNN_TIMER(fwd_t);

  STOP_CNN_TIMER(fwd_t);
}

void train_bwd_branch_layer(branch_layer *l)
{
  START_CNN_TIMER(bwd_t);

  chkCUDNN(cudnnOpTensor(
        l->cudnn, l->op_desc, 
        &one, l->d_output_desc, l->d_output[0],
        &one, l->d_output_desc, l->d_output[1],
        &zero, l->d_input_desc, l->d_input));

  for (int i = 2; i < l->out_cnt; i++) {
    chkCUDNN(cudnnAddTensor(
          l->cudnn, &one, l->d_output_desc, l->d_output[i],
          &one, l->d_input_desc, l->d_input));
  }

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
