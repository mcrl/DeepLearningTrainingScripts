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

void init_softmax_layer(
    softmax_layer *l, const char *name, int batch_size, int out)
{
  ////////////////////////////////////////////////////////////////
  // 1. Initialize Parameters
  ////////////////////////////////////////////////////////////////
  strcpy(l->name, name);

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
  create_buffer_data(
      &l->input, CUDNN_DATA_FLOAT, 4, l->batch_size, l->out, 1, 1);

  create_buffer_data_gradient(
      &l->d_input, CUDNN_DATA_FLOAT, 4, l->batch_size, l->out, 1, 1);

  create_buffer_data(
      &l->output, CUDNN_DATA_FLOAT, 4, l->batch_size, l->out, 1, 1);

  create_buffer_data_gradient(
      &l->d_output, CUDNN_DATA_FLOAT, 4, l->batch_size, l->out, 1, 1);

  create_buffer_data(
      &l->label, CUDNN_DATA_INT32, 4, l->batch_size, 1, 1, 1);
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

void set_label(softmax_layer *l, int *label_in)
{
  static bool initialized = false;

  if (!initialized) {
    alloc_buffer(l->label);
    alloc_buffer(l->d_output);
    initialized = true;
  }

  write_buffer(l->label, label_in, true);
  execute_set_label(l->label, l->d_output);
  synch_device();
}

float get_loss(softmax_layer *l, int *label_in)
{
  size_t size = logical_buffer_size(l->output);
  float *result = (float *)malloc(size);
  read_buffer(result, l->output, true);

  float sum = 0;
  for (int i = 0; i < l->batch_size; i++) {
    float *cur = result + l->out * i;
    int ans = label_in[i];
    float loss = log(cur[ans]);
    printf("%d, %f, %f\n", ans, cur[ans], loss);
    sum -= loss;
  }

  return sum / l->batch_size;
}

void print_time_softmax_layer(softmax_layer *l)
{
  printf("%s, %.3f, %.3f, %.3f, %.3f\n",
      l->name, l->fwd_t, l->bwd_t, 0.0f, 0.0f);
}

void clear_time_softmax_layer(softmax_layer *l)
{
  l->fwd_t = 0.0;
  l->bwd_t = 0.0;
}
