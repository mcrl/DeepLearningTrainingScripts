#include <math.h>
#include <time.h>
#include <string.h>

#include <cuda_runtime_api.h>
#include <cuda.h>
#include <cudnn.h>
#include <mpi.h>

#include "layer.h"
#include "params.h"
#include "utils.h"
#include "memory.h"
#include "execute.h"

// FIXME: remove dependencies
extern int num_nodes;
extern int node_id;

void init_softmax_layer(
    softmax_layer *l, const char *name,
    int batch_size, int out, softmax_type type)
{
  ////////////////////////////////////////////////////////////////
  // 1. Initialize Parameters
  ////////////////////////////////////////////////////////////////
  strcpy(l->name, name);

  l->batch_size = batch_size;
  l->out = out;

  l->type = type;

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
      (l->type == LOG_T) ? CUDNN_SOFTMAX_LOG : CUDNN_SOFTMAX_ACCURATE,
      CUDNN_SOFTMAX_MODE_CHANNEL, l->input, l->output);
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
}

float get_loss(softmax_layer *l, int *label_in)
{
  size_t local_size, global_size;
  logical_buffer_size(l->output, &local_size, &global_size);

  float *result = (float *)malloc(global_size);
  read_buffer(result, l->output, true);

  // FIXME: rearrange below statements to a subroutine
  int *recvcounts = (int *)malloc(num_nodes * sizeof(int));
  int *displs = (int *)malloc(num_nodes * sizeof(int));

  MPI_Gather(
      &local_size, sizeof(int), MPI_BYTE, recvcounts,
      sizeof(int), MPI_BYTE, 0, MPI_COMM_WORLD);

  displs[0] = 0;
  for (int i = 1; i < num_nodes; ++i) {
    displs[i] = displs[i - 1] + recvcounts[i - 1];
  }

  if (node_id == 0) {
    MPI_Gatherv(
        MPI_IN_PLACE, local_size, MPI_BYTE,
        result, recvcounts, displs, MPI_BYTE, 0, MPI_COMM_WORLD);
  }
  else {
    void *srcbuf = (void *)((char *)result + l->output->offset_in_bytes[0]);
    MPI_Gatherv(
        srcbuf, local_size, MPI_BYTE,
        NULL, NULL, NULL, MPI_BYTE, 0, MPI_COMM_WORLD);
  }
  // FIXME: rearrange above statements to a subroutine

  float sum = 0;
  for (int i = 0; i < l->batch_size; i++) {
    float *cur = result + l->out * i;
    int ans = label_in[i];
    float loss = log(cur[ans]);
    sum -= loss;

    if (node_id == 0) {
      printf("%d, %f, %f\n", ans, cur[ans], loss);
    }
  }

  free(recvcounts);
  free(displs);
  free(result);

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
