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

void init_act_layer(
    act_layer *l, cudnnHandle_t cudnn,
    int batch, int channel, int height, int width)
{
  l->cudnn = cudnn;
  l->width = width;
  l->height = height;
  l->channel = channel;
  l->type = relu;

  l->input = NULL;
  l->d_output = NULL;

  l->fwd_t = 0;
  l->bwd_t = 0;

  chkCUDNN(cudnnCreateTensorDescriptor(&l->input_desc));
  chkCUDNN(cudnnCreateTensorDescriptor(&l->d_input_desc));
  chkCUDNN(cudnnCreateTensorDescriptor(&l->output_desc));
  chkCUDNN(cudnnCreateTensorDescriptor(&l->d_output_desc));

  chkCUDNN(cudnnCreateActivationDescriptor(&l->act_desc));

  switch (l->type) {
    case relu:
      chkCUDNN(cudnnSetActivationDescriptor(
            l->act_desc, CUDNN_ACTIVATION_RELU, CUDNN_NOT_PROPAGATE_NAN, 20.0));
      break;

    default:
      assert(0);
      break;
  }

  MALLOC_TENSOR_FLOAT(&l->output, batch, channel, height, width);
  MALLOC_TENSOR_FLOAT(&l->d_input, batch, channel, height, width);

  chkCUDNN(cudnnSetTensor4dDescriptor(
        l->input_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
        batch, channel, height, width));

  chkCUDNN(cudnnSetTensor4dDescriptor(
        l->d_input_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
        batch, channel, height, width));

  chkCUDNN(cudnnSetTensor4dDescriptor(
        l->output_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
        batch, channel, height, width));

  chkCUDNN(cudnnSetTensor4dDescriptor(
        l->d_output_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
        batch, channel, height, width));
}

void train_fwd_act_layer(act_layer *l)
{
  START_CNN_TIMER(fwd_t);

  chkCUDNN(cudnnActivationForward(
        l->cudnn, l->act_desc,
        &one, l->input_desc, l->input,
        &zero, l->output_desc, l->output));

  STOP_CNN_TIMER(fwd_t);
}

void train_bwd_act_layer(act_layer *l)
{
  START_CNN_TIMER(bwd_t);

  chkCUDNN(cudnnActivationBackward(
        l->cudnn, l->act_desc,
        &one, l->output_desc, l->output,
        l->d_output_desc, l->d_output, l->input_desc, l->input,
        &zero, l->d_input_desc, l->d_input));

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
