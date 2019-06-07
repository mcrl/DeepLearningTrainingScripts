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

size_t workspace_bytes = 0;
float *workspace = NULL;

void init_conv_layer(
    conv_layer *l, cudnnHandle_t cudnn,
    int batch_size,
    int filter_height, int filter_width,
    int pad_height, int pad_width,
    int stride_x, int stride_y,
    int in, int out, int height, int width)
{
  size_t tmp;

  l->cudnn = cudnn;
  l->filter_height = filter_height;
  l->filter_width = filter_width;
  l->pad_height = pad_height;
  l->pad_width = pad_width;
  l->stride_x = stride_x;
  l->stride_y = stride_y;
  l->channels_out = out;
  l->channels_in = in;

  l->input = NULL;
  l->d_output = NULL;

  l->fwd_t = 0;
  l->bwd_data_t = 0;
  l->bwd_filter_t = 0;
  l->bwd_update_t = 0;

  chkCUDNN(cudnnCreateTensorDescriptor(&l->input_desc));
  chkCUDNN(cudnnCreateTensorDescriptor(&l->output_desc));

  chkCUDNN(cudnnCreateTensorDescriptor(&l->d_input_desc));
  chkCUDNN(cudnnCreateTensorDescriptor(&l->d_output_desc));

  chkCUDNN(cudnnCreateFilterDescriptor(&l->filter_desc));
  chkCUDNN(cudnnCreateFilterDescriptor(&l->d_filter_desc));

  chkCUDNN(cudnnCreateConvolutionDescriptor(&l->conv_desc));

  chkCUDNN(cudnnSetFilter4dDescriptor(
        l->filter_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
        l->channels_out, l->channels_in, l->filter_height, l->filter_width));
  
  chkCUDNN(cudnnSetFilter4dDescriptor(
        l->d_filter_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
        l->channels_out, l->channels_in, l->filter_height, l->filter_width));

  chkCUDNN(cudnnSetConvolution2dDescriptor(
        l->conv_desc, l->pad_height, l->pad_width, l->stride_x, l->stride_y, 1, 1,
        CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT));

  const int height_output =
    CALC_SIZE(height, l->filter_height, l->pad_height, l->stride_x);

  const int width_output =
    CALC_SIZE(width, l->filter_width, l->pad_width, l->stride_y);

  chkCUDNN(cudnnSetTensor4dDescriptor(
        l->input_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
        batch_size, l->channels_in, height, width));

  chkCUDNN(cudnnSetTensor4dDescriptor(
        l->output_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
        batch_size, l->channels_out, height_output, width_output));

  chkCUDNN(cudnnSetTensor4dDescriptor(
        l->d_input_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
        batch_size, l->channels_in, height, width));

  chkCUDNN(cudnnSetTensor4dDescriptor(
        l->d_output_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
        batch_size, l->channels_out, height_output, width_output));

  chkCUDNN(cudnnGetConvolutionForwardAlgorithm(
        l->cudnn, l->input_desc, l->filter_desc, l->conv_desc, l->output_desc,
        CUDNN_CONVOLUTION_FWD_PREFER_FASTEST, 0, &l->fwd_algo));

  chkCUDNN(cudnnGetConvolutionForwardWorkspaceSize(
        l->cudnn, l->input_desc, l->filter_desc,
        l->conv_desc, l->output_desc, l->fwd_algo, &tmp));

  workspace_bytes = (workspace_bytes > tmp) ? workspace_bytes : tmp; 
  l->conv_workspace_bytes = tmp;

  chkCUDNN(cudnnGetConvolutionBackwardDataAlgorithm(
        l->cudnn, l->filter_desc, l->d_output_desc, l->conv_desc, l->d_input_desc,
        CUDNN_CONVOLUTION_BWD_DATA_PREFER_FASTEST, 0, &l->bwd_data_algo));

  chkCUDNN(cudnnGetConvolutionBackwardDataWorkspaceSize(
        l->cudnn, l->filter_desc, l->d_output_desc,
        l->conv_desc, l->d_input_desc, l->bwd_data_algo, &tmp));

  workspace_bytes = (workspace_bytes > tmp) ? workspace_bytes : tmp; 
  l->conv_workspace_bwd_data_bytes = tmp;

  chkCUDNN(cudnnGetConvolutionBackwardFilterAlgorithm(
        l->cudnn, l->input_desc, l->d_output_desc, l->conv_desc, l->d_filter_desc,
        CUDNN_CONVOLUTION_BWD_FILTER_PREFER_FASTEST, 0, &l->bwd_filter_algo));

#ifdef DET_CUDNN
    l->bwd_filter_algo = CUDNN_CONVOLUTION_BWD_FILTER_ALGO_1;
#endif

  chkCUDNN(cudnnGetConvolutionBackwardFilterWorkspaceSize(
        l->cudnn, l->input_desc, l->d_output_desc, l->conv_desc,
        l->d_filter_desc, l->bwd_filter_algo, &tmp));

  workspace_bytes = (workspace_bytes > tmp) ? workspace_bytes : tmp; 
  l->conv_workspace_bwd_filter_bytes = tmp;

  MALLOC_TENSOR_FLOAT(&l->output, batch_size, l->channels_out, height_output, width_output);
  MALLOC_TENSOR_FLOAT(&l->d_input, batch_size, l->channels_in, height, width);
  MALLOC_TENSOR_FLOAT(&l->filter, out, in, filter_height, filter_width);
  MALLOC_TENSOR_FLOAT(&l->d_filter, out, in, filter_height, filter_width);
}

void init_conv_workspace()
{
  chkCUDA(cudaMalloc((void **)&workspace, workspace_bytes));
}

int set_conv_filter(conv_layer l, float *filter)
{
  size_t s = PSIZE_CONV(l);
  chkCUDA(cudaMemcpy(l.filter, filter, s, cudaMemcpyHostToDevice));
  return s / sizeof(float);
}

int get_conv_filter(conv_layer l, float *filter)
{
  size_t s = PSIZE_CONV(l);
  chkCUDA(cudaMemcpy(filter, l.filter, s, cudaMemcpyDeviceToHost));
  return s / sizeof(float);
}

void train_fwd_conv_layer(conv_layer *l)
{
  START_CNN_TIMER(fwd_t);

  chkCUDNN(cudnnConvolutionForward(
        l->cudnn, &one, l->input_desc, l->input,
        l->filter_desc, l->filter, l->conv_desc,
        l->fwd_algo, workspace, l->conv_workspace_bytes,
        &zero, l->output_desc, l->output));

  STOP_CNN_TIMER(fwd_t);
}

void train_bwd_conv_layer(conv_layer *l)
{
  START_CNN_TIMER(bwd_data_t);

  chkCUDNN(cudnnConvolutionBackwardData(
        l->cudnn, &one, l->filter_desc, l->filter,
        l->d_output_desc, l->d_output, l->conv_desc,
        l->bwd_data_algo, workspace, l->conv_workspace_bwd_data_bytes,
        &zero, l->d_input_desc, l->d_input));

  STOP_CNN_TIMER(bwd_data_t);

  START_CNN_TIMER(bwd_filter_t);

  chkCUDNN(cudnnConvolutionBackwardFilter(
        l->cudnn, &one, l->input_desc, l->input,
        l->d_output_desc, l->d_output, l->conv_desc,
        l->bwd_filter_algo, workspace, l->conv_workspace_bwd_filter_bytes,
        &zero, l->d_filter_desc, l->d_filter));

  STOP_CNN_TIMER(bwd_filter_t);

  START_CNN_TIMER(bwd_update_t);

  cublas_apply_grad(
      l->filter, l->d_filter, params.learning_rate,
      l->channels_in * l->channels_out * l->filter_width * l->filter_height);

  STOP_CNN_TIMER(bwd_update_t);
}

void print_time_conv_layer(conv_layer *l, char *name)
{
  printf("%s, %.3f, %.3f, %.3f, %.3f\n",
      name, l->fwd_t, l->bwd_data_t, l->bwd_filter_t, l->bwd_update_t);
}

void clear_time_conv_layer(conv_layer *l)
{
  l->fwd_t = 0.0;
  l->bwd_data_t = 0.0;
  l->bwd_filter_t = 0.0;
  l->bwd_update_t = 0.0;
}
