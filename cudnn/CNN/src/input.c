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

void init_input_layer(
    input_layer *l, const char *name,
    int batch_size, int channel, int height, int width)
{
  ////////////////////////////////////////////////////////////////
  // 1. Initialize Parameters
  ////////////////////////////////////////////////////////////////
  strcpy(l->name, name);

  l->batch_size = batch_size;
  l->channel = channel;
  l->height = height;
  l->width = width;

  l->output = NULL;
  l->d_output = NULL;

  ////////////////////////////////////////////////////////////////
  // 2. Create Tensors
  ////////////////////////////////////////////////////////////////
  create_buffer_data(
      &l->output, CUDNN_DATA_FLOAT, 4,
      l->batch_size, l->channel, l->height, l->width);

  create_buffer_data_gradient(
      &l->d_output, CUDNN_DATA_FLOAT, 4,
      l->batch_size, l->channel, l->height, l->width);
}

void set_input(input_layer *l, float *data_in)
{
  write_buffer(l->output, data_in, true);
}
