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

extern int num_devices;

static unsigned long long get_seed()
{
  return 2019ull;
}

void init_dropout_layer(
    dropout_layer *l, const char *name,
    int batch_size, int out, float rate)
{
  size_t rs_size;
  size_t st_size;

  ////////////////////////////////////////////////////////////////
  // 1. Initialize Parameters
  ////////////////////////////////////////////////////////////////
  strcpy(l->name, name);

  l->batch_size = batch_size;
  l->out = out;

  l->rate = rate;
  l->seed = get_seed();

  l->input = NULL;
  l->d_input = NULL;

  l->output = NULL;
  l->d_output = NULL;

  l->rs = NULL;
  l->st = NULL;

  clear_time_dropout_layer(l);

  ////////////////////////////////////////////////////////////////
  // 2. Set Dropout Descriptor
  ////////////////////////////////////////////////////////////////
  for (int dev = 0; dev < num_devices; dev++) {
    chkCUDNN(cudnnCreateDropoutDescriptor(&l->dr_desc[dev]));
  }

  execute_get_dropout_st_size(&st_size);

  create_buffer_reserve_space(&l->st, st_size);

  execute_set_dropout(l->dr_desc, l->rate, l->seed, l->st);

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

  ////////////////////////////////////////////////////////////////
  // 4. Create Reserve Space & State
  ////////////////////////////////////////////////////////////////
  execute_get_dropout_rs_size(l->input, &rs_size);

  create_buffer_reserve_space(&l->rs, rs_size);
}

void train_fwd_dropout_layer(dropout_layer *l)
{
  START_CNN_TIMER(fwd_t);
  execute_dropout_fwd(l->dr_desc, l->input, l->output, l->rs);
  STOP_CNN_TIMER(fwd_t);
}

void train_bwd_dropout_layer(dropout_layer *l)
{
  START_CNN_TIMER(bwd_t);
  execute_dropout_bwd(l->dr_desc, l->d_output, l->d_input, l->rs);
  STOP_CNN_TIMER(bwd_t);
}

void print_time_dropout_layer(dropout_layer *l)
{
  printf("%s, %.3f, %.3f, %.3f, %.3f\n",
      l->name, l->fwd_t, l->bwd_t, 0.0f, 0.0f);
}

void clear_time_dropout_layer(dropout_layer *l)
{
  l->fwd_t = 0.0;
  l->bwd_t = 0.0;
}
