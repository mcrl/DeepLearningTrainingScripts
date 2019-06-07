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

void init_dropout_layer(
    dropout_layer *l, cudnnHandle_t cudnn,
    int batch, int out, float dr_rate)
{
  l->cudnn = cudnn;
  l->out = width;

  l->output_desc = NULL;
  l->d_output_desc = NULL;

  l->fwd_t = 0;
  l->bwd_t = 0;

  chkCUDNN(cudnnCreateTensorDescriptor(&l->input_desc));
  chkCUDNN(cudnnCreateTensorDescriptor(&l->d_input_desc));
  chkCUDNN(cudnnCreateTensorDescriptor(&l->output_desc));
  chkCUDNN(cudnnCreateTensorDescriptor(&l->d_output_desc));

  chkCUDNN(cudnnCreateDropoutDescriptor(&l->dropout_desc));

  MALLOC_TENSOR_FLOAT(&l->d_input, batch, 1, 1, out);
  MALLOC_TENSOR_FLOAT(&l->output, batch, 1, 1, out);

  chkCUDNN(cudnnSetTensor4dDescriptor(
        l->output_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
        batch, 1, 1, out));

  chkCUDNN(cudnnSetTensor4dDescriptor(
        l->d_output_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
        batch, 1, 1, out));

  chkCUDNN(cudnnSetTensor4dDescriptor(
        l->input_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
        batch, 1, 1, out));

  chkCUDNN(cudnnSetTensor4dDescriptor(
        l->d_input_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
        batch, 1, 1, out));
}
