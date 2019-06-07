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

void init_pool_layer(
    pool_layer *l, cudnnHandle_t cudnn,
    int batch_size,
    int filter_height, int filter_width,
    int pad_height, int pad_width,
    int stride_x, int stride_y,
    int channel, int height, int width, POOL_TYPE type)
{
  l->cudnn = cudnn;
  l->filter_height = filter_height;
  l->filter_width = filter_width;
  l->stride_x = stride_x;
  l->stride_y = stride_y;
  l->pad_height = pad_height;
  l->pad_width = pad_width;
  l->channel = channel;
  l->type = type;

  l->input = NULL;
  l->d_output = NULL;

  l->fwd_t = 0;
  l->bwd_t = 0;

  chkCUDNN(cudnnCreateTensorDescriptor(&l->input_desc));
  chkCUDNN(cudnnCreateTensorDescriptor(&l->d_input_desc));
  chkCUDNN(cudnnCreateTensorDescriptor(&l->output_desc));
  chkCUDNN(cudnnCreateTensorDescriptor(&l->d_output_desc));

  chkCUDNN(cudnnCreatePoolingDescriptor(&l->pooling_desc));

  const int height_output =
    CALC_SIZE(height, l->filter_height, l->pad_height, l->stride_x);

  const int width_output =
    CALC_SIZE(width, l->filter_width, l->pad_width, l->stride_y);

  l->height = height;
  l->width = width;
  l->height_output = height_output;
  l->width_output = width_output;
 
  MALLOC_TENSOR_FLOAT(&l->output, batch_size, l->channel, height_output, width_output);
  MALLOC_TENSOR_FLOAT(&l->d_input, batch_size, l->channel, height, width);

  chkCUDNN(cudnnSetTensor4dDescriptor(
        l->input_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
        batch_size, l->channel, height, width));

  chkCUDNN(cudnnSetTensor4dDescriptor(
        l->d_input_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
        batch_size, l->channel, height, width));

  chkCUDNN(cudnnSetTensor4dDescriptor(
        l->output_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
        batch_size, l->channel, height_output, width_output));

  chkCUDNN(cudnnSetTensor4dDescriptor(
        l->d_output_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
        batch_size, l->channel, height_output, width_output));

  chkCUDNN(cudnnSetPooling2dDescriptor(
        l->pooling_desc,
        (l->type == max) ? CUDNN_POOLING_MAX_DETERMINISTIC : CUDNN_POOLING_AVERAGE_COUNT_EXCLUDE_PADDING,
        CUDNN_NOT_PROPAGATE_NAN,
        filter_height, filter_width,
        pad_height, pad_width,
        stride_x, stride_y));
}

void train_fwd_pool_layer(pool_layer *l)
{
  START_CNN_TIMER(fwd_t);

  chkCUDNN(cudnnPoolingForward(
        l->cudnn, l->pooling_desc,
        &one, l->input_desc, l->input,
        &zero, l->output_desc, l->output));

  STOP_CNN_TIMER(fwd_t);
}

void train_bwd_pool_layer(pool_layer *l)
{
  START_CNN_TIMER(bwd_t);

  chkCUDNN(cudnnPoolingBackward(
        l->cudnn, l->pooling_desc,
        &one, l->output_desc, l->output,
        l->d_output_desc, l->d_output, l->input_desc, l->input,
        &zero, l->d_input_desc, l->d_input));

  STOP_CNN_TIMER(bwd_t);
}

void print_time_pool_layer(pool_layer *l, char *name)
{
  printf("%s, %.3f, %.3f, %.3f, %.3f\n",
      name, l->fwd_t, l->bwd_t, 0.0f, 0.0f);
}

void clear_time_pool_layer(pool_layer *l)
{
  l->fwd_t = 0.0;
  l->bwd_t = 0.0;
}
