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

extern size_t workspace_bytes;
extern float *workspace;

static float one = 1.0f;
static float zero = 0.0f;

void init_fc_layer(
    fc_layer *l, cudnnHandle_t cudnn,
    int batch_size, int in, int out)
{
  size_t tmp;

  l->cudnn = cudnn;
  l->batch_size = batch_size;
  l->in = in;
  l->out = out;

  l->input = NULL;
  l->d_output = NULL;

  l->fwd_t = 0;
  l->bwd_data_t = 0;
  l->bwd_weight_t = 0;
  l->bwd_update_t = 0;

  chkCUDNN(cudnnCreateTensorDescriptor(&l->input_desc));
  chkCUDNN(cudnnCreateTensorDescriptor(&l->output_desc));

  chkCUDNN(cudnnCreateTensorDescriptor(&l->d_input_desc));
  chkCUDNN(cudnnCreateTensorDescriptor(&l->d_output_desc));

  chkCUDNN(cudnnCreateFilterDescriptor(&l->filter_desc));
  chkCUDNN(cudnnCreateFilterDescriptor(&l->d_filter_desc));

  chkCUDNN(cudnnCreateConvolutionDescriptor(&l->conv_desc));

  chkCUDNN(cudnnSetTensor4dDescriptor(
        l->input_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
        batch_size, in, 1, 1));

  chkCUDNN(cudnnSetTensor4dDescriptor(
        l->d_input_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
        batch_size, in, 1, 1));

  chkCUDNN(cudnnSetTensor4dDescriptor(
        l->output_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
        batch_size, out, 1, 1));

  chkCUDNN(cudnnSetTensor4dDescriptor(
        l->d_output_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
        batch_size, out, 1, 1));

  chkCUDNN(cudnnSetFilter4dDescriptor(
        l->filter_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
        out, in, 1, 1));

  chkCUDNN(cudnnSetFilter4dDescriptor(
        l->d_filter_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
        out, in, 1, 1));

  chkCUDNN(cudnnSetConvolution2dDescriptor(
        l->conv_desc, 0, 0, 1, 1, 1, 1,
        CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT));

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
        l->cudnn, l->input_desc, l->d_output_desc,
        l->conv_desc, l->d_filter_desc, l->bwd_filter_algo, &tmp));

  workspace_bytes = (workspace_bytes > tmp) ? workspace_bytes : tmp; 
  l->conv_workspace_bwd_filter_bytes = tmp;

  MALLOC_TENSOR_FLOAT(&l->filter, out, in, 1, 1);
  MALLOC_TENSOR_FLOAT(&l->d_filter, out, in, 1, 1);
  MALLOC_TENSOR_FLOAT(&l->output, batch_size, out, 1, 1);
  MALLOC_TENSOR_FLOAT(&l->d_input, batch_size, in, 1, 1);
}

void train_fwd_fc_layer(fc_layer *l)
{
  START_CNN_TIMER(fwd_t);

  chkCUDNN(cudnnConvolutionForward(
        l->cudnn, &one, l->input_desc, l->input, l->filter_desc, l->filter,
        l->conv_desc, l->fwd_algo, workspace, l->conv_workspace_bytes,
        &zero, l->output_desc, l->output));

  STOP_CNN_TIMER(fwd_t);
}

void train_bwd_fc_layer(fc_layer *l)
{
  START_CNN_TIMER(bwd_data_t);

  chkCUDNN(cudnnConvolutionBackwardData(l->cudnn,
    &one, l->filter_desc, l->filter,
    l->d_output_desc, l->d_output, l->conv_desc,
    l->bwd_data_algo,
    workspace, l->conv_workspace_bwd_data_bytes,
    &zero, l->d_input_desc, l->d_input));

  STOP_CNN_TIMER(bwd_data_t);

  START_CNN_TIMER(bwd_weight_t);

  chkCUDNN(cudnnConvolutionBackwardFilter(l->cudnn,
    &one, l->input_desc, l->input,
    l->d_output_desc, l->d_output, l->conv_desc,
    l->bwd_filter_algo,
    workspace, l->conv_workspace_bwd_filter_bytes,
    &zero, l->d_filter_desc, l->d_filter));

  STOP_CNN_TIMER(bwd_weight_t);

  START_CNN_TIMER(bwd_update_t);

  cublas_apply_grad(l->filter, l->d_filter, params.learning_rate,  l->out * l->in);

  STOP_CNN_TIMER(bwd_update_t);
}

int set_fc_weight(fc_layer l, float *weight)
{
  size_t s = PSIZE_FC(l);
  chkCUDA(cudaMemcpy(l.filter, weight, s, cudaMemcpyHostToDevice));
  return s / sizeof(float);
}

int get_fc_weight(fc_layer l, float *weight)
{
  size_t s = PSIZE_FC(l);
  chkCUDA(cudaMemcpy(weight, l.filter, s, cudaMemcpyDeviceToHost));
  return s / sizeof(float);
}

void print_time_fc_layer(fc_layer *l, char *name)
{
  printf("%s, %.3f, %.3f, %.3f, %.3f\n",
      name, l->fwd_t, l->bwd_data_t, l->bwd_weight_t, l->bwd_update_t);
}

void clear_time_fc_layer(fc_layer *l)
{
  l->fwd_t = 0.0;
  l->bwd_data_t = 0.0;
  l->bwd_weight_t = 0.0;
  l->bwd_update_t = 0.0;
}
