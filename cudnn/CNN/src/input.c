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

void init_input_layer(
    input_layer *l, cudnnHandle_t cudnn,
    int batch, int channel, int height, int width)
{
  l->cudnn = cudnn;
  l->width = width;
  l->height = height;
  l->channel = channel;

  l->d_output = NULL;

  chkCUDNN(cudnnCreateTensorDescriptor(&l->output_desc));
  chkCUDNN(cudnnCreateTensorDescriptor(&l->d_output_desc));

  chkCUDNN(cudnnSetTensor4dDescriptor(
        l->output_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
        batch, l->channel, height, width));

  chkCUDNN(cudnnSetTensor4dDescriptor(
        l->d_output_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
        batch, l->channel, height, width));

  MALLOC_TENSOR_FLOAT(&l->output, batch, channel, height, width);
}

