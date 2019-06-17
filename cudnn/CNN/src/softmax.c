#include <math.h>
#include <time.h>

#include <builtin_types.h>
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

void init_softmax_layer(softmax_layer *l, int batch_size, int out)
{
  ////////////////////////////////////////////////////////////////
  // 1. Initialize Parameters
  ////////////////////////////////////////////////////////////////
  l->batch_size = batch_size;
  l->out = out;

  l->label = NULL;

  l->input = NULL;
  l->d_input = NULL;

  l->output = NULL;
  l->d_output = NULL;

  clear_time_softmax_layer(l);

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
      &l->input, 4, CUDNN_DATA_FLOAT, l->batch_size, l->out, 1, 1);

  create_buffer[DATA_GRADIENT](
      &l->d_input, 4, CUDNN_DATA_FLOAT, l->batch_size, l->out, 1, 1);

  create_buffer[DATA](
      &l->output, 4, CUDNN_DATA_FLOAT, l->batch_size, l->out, 1, 1);

  create_buffer[DATA_GRADIENT](
      &l->d_output, 4, CUDNN_DATA_FLOAT, l->batch_size, l->out, 1, 1);

  create_buffer[DATA](
      &l->label, 4, CUDNN_DATA_INT32, l->batch_size, l->out, 1, 1);
}

void train_fwd_softmax_layer(softmax_layer *l)
{
  START_CNN_TIMER(fwd_t);
  execute_softmax_fwd(
      CUDNN_SOFTMAX_ACCURATE, CUDNN_SOFTMAX_MODE_CHANNEL, l->input, l->output);
  STOP_CNN_TIMER(fwd_t);
}

void train_bwd_softmax_layer(softmax_layer *l)
{
  START_CNN_TIMER(bwd_t);
  execute_elt(l->op_desc, l->output, l->d_output, l->d_input);
  STOP_CNN_TIMER(bwd_t);
}

float get_loss(softmax_layer *l, int *label_in)
{
  /*
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
  */
  return 0;
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
