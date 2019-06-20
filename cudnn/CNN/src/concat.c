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

void init_concat_layer(
    concat_layer *l, const char *name,
    int batch_size, int fan_in, int input_channel[], int height, int width)
{
  ////////////////////////////////////////////////////////////////
  // 1. Initialize Parameters
  ////////////////////////////////////////////////////////////////
  strcpy(l->name, name);

  l->batch_size = batch_size;
  l->output_channel = 0;
  l->width = width;
  l->height = height;

  l->fan_in = fan_in;

  for (int i = 0; i < l->fan_in; i++) {
    l->input_channel[i] = input_channel[i];
    l->output_channel += input_channel[i];

    l->input[i] = NULL;
    l->d_input[i] = NULL;
  }

  l->output = NULL;
  l->d_output = NULL;

  clear_time_concat_layer(l);

  ////////////////////////////////////////////////////////////////
  // 2. Create Tensors
  ////////////////////////////////////////////////////////////////
  for (int i = 0; i < l->fan_in; i++) {
    create_buffer[DATA](
        &l->input[i], 4, CUDNN_DATA_FLOAT, l->batch_size,
        l->input_channel[i], l->height, l->width);

    create_buffer[DATA_GRADIENT](
        &l->d_input[i], 4, CUDNN_DATA_FLOAT, l->batch_size,
        l->input_channel[i], l->height, l->width);
  }

  create_buffer[DATA](
      &l->output, 4, CUDNN_DATA_FLOAT, l->batch_size,
      l->output_channel, l->height, l->width);

  create_buffer[DATA_GRADIENT](
      &l->d_output, 4, CUDNN_DATA_FLOAT, l->batch_size,
      l->output_channel, l->height, l->width);
}

void train_fwd_concat_layer(concat_layer *l)
{
  START_CNN_TIMER(fwd_t);
  execute_concat_fwd(l->fan_in, l->input, l->output);
  STOP_CNN_TIMER(fwd_t);
}

void train_bwd_concat_layer(concat_layer *l)
{
  START_CNN_TIMER(bwd_t);
  execute_concat_bwd(l->fan_in, l->d_output, l->d_input);
  STOP_CNN_TIMER(bwd_t);
}

void print_time_concat_layer(concat_layer *l)
{
  printf("%s, %.3f, %.3f, %.3f, %.3f\n",
      l->name, l->fwd_t, l->bwd_t, 0.0f, 0.0f);
}

void clear_time_concat_layer(concat_layer *l)
{
  l->fwd_t = 0.0;
  l->bwd_t = 0.0;
}
