#include <math.h>

#include <builtin_types.h>
#include <cuda_runtime_api.h>
#include <cuda.h>
#include <cudnn.h>

#include "params.h"
#include "utils.h"
#include "layer.h"

static float one = 1.0f;
static float zero = 0.0f;

void init_softmax_layer(
    softmax_layer *l, cudnnHandle_t cudnn,
    int batch, int out)
{
  l->cudnn = cudnn;
  l->out = out;
  l->batch_size = batch;

  l->fwd_t = 0;
  l->bwd_t = 0;

  chkCUDNN(cudnnCreateTensorDescriptor(&l->input_desc));
  chkCUDNN(cudnnCreateTensorDescriptor(&l->d_input_desc));
  chkCUDNN(cudnnCreateTensorDescriptor(&l->output_desc));
  chkCUDNN(cudnnCreateOpTensorDescriptor(&l->op_desc));

  chkCUDNN(cudnnSetTensor4dDescriptor(
        l->d_input_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
        batch, l->out, 1, 1));

  chkCUDNN(cudnnSetTensor4dDescriptor(
        l->input_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
        batch, l->out, 1, 1));

  chkCUDNN(cudnnSetTensor4dDescriptor(
        l->output_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
        batch, l->out, 1, 1));

  chkCUDNN(cudnnSetOpTensorDescriptor(
        l->op_desc, CUDNN_OP_TENSOR_ADD, CUDNN_DATA_FLOAT, CUDNN_NOT_PROPAGATE_NAN));

  MALLOC_TENSOR_FLOAT(&l->output, batch, 1, 1, l->out);
  MALLOC_TENSOR_FLOAT(&l->d_input, batch, 1, 1, l->out);
  MALLOC_TENSOR_FLOAT(&l->label, batch, 1, 1, l->out);
  MALLOC_TENSOR_INT(&l->label_in, batch, 1, 1, l->out);
}

void train_fwd_softmax_layer(softmax_layer *l)
{
  START_CNN_TIMER(fwd_t);

  chkCUDNN(cudnnSoftmaxForward(
        l->cudnn, CUDNN_SOFTMAX_ACCURATE, CUDNN_SOFTMAX_MODE_CHANNEL,
        &one, l->input_desc, l->input,
        &zero, l->output_desc, l->output));

  STOP_CNN_TIMER(fwd_t);
}

void train_bwd_softmax_layer(softmax_layer *l)
{
  START_CNN_TIMER(bwd_t);

  chkCUDNN(cudnnOpTensor(
        l->cudnn, l->op_desc, 
        &one, l->output_desc, l->output,
        &one, l->output_desc, l->label,
        &zero, l->d_input_desc, l->d_input));

  STOP_CNN_TIMER(bwd_t);
}

float get_loss(softmax_layer *l, int *label_in)
{
  float *result = (float *)malloc(sizeof(float) * l->out * l->batch_size);
  chkCUDA(cudaMemcpy(result, l->output, sizeof(float) * l->out * l->batch_size, cudaMemcpyDeviceToHost));

  float sum = 0;
  for (int i = 0; i < l->batch_size; i++) {
    float *cur = result + l->out * i;
    int ans = label_in[i];
    float loss = log(cur[ans]);
    printf("%d, %f, %f\n", ans, cur[ans], loss);
    sum -= loss;
  }

  return (sum / l->batch_size);
}

void print_time_softmax_layer(softmax_layer *l, char *name)
{
  printf("%s, %.3f, %.3f, %.3f, %.3f\n",
      name, l->fwd_t, l->bwd_t, 0.0f, 0.0f);
}

void clear_time_softmax_layer(softmax_layer *l)
{
  l->fwd_t = 0.0;
  l->bwd_t = 0.0;
}
