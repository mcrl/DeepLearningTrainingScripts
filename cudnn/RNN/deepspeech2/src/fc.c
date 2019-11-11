#include <math.h>

#include <builtin_types.h>
#include <cuda_runtime_api.h>
#include <cuda.h>
#include <cudnn.h>

#include "deepspeech.h"
#include "deepspeech_cuda.h"
#include "fc.h"
#include "params.h"
#include "utils.h"

void init_fc_layer(fc_layer *fcl, cudnnHandle_t cudnn,
  int in_features, int out_features,
  float *filter, float *bn_scale, float *bn_bias, int *off_d, int max_batch)
{
  fcl->cudnn = cudnn;
  fcl->in_features = in_features;
  fcl->out_features = out_features;

  chkCUDNN(cudnnCreateTensorDescriptor(&fcl->bn_input_desc));
  chkCUDNN(cudnnCreateTensorDescriptor(&fcl->bn_output_desc));

  chkCUDNN(cudnnCreateTensorDescriptor(&fcl->input_desc));
  chkCUDNN(cudnnCreateTensorDescriptor(&fcl->after_bn_desc));
  chkCUDNN(cudnnCreateTensorDescriptor(&fcl->after_conv_desc));
  chkCUDNN(cudnnCreateTensorDescriptor(&fcl->before_softmax_desc));
  chkCUDNN(cudnnCreateTensorDescriptor(&fcl->after_softmax_desc));

  chkCUDNN(cudnnCreateTensorDescriptor(&fcl->d_bn_input_desc));
  chkCUDNN(cudnnCreateTensorDescriptor(&fcl->d_bn_output_desc));

  chkCUDNN(cudnnCreateTensorDescriptor(&fcl->d_input_desc));
  chkCUDNN(cudnnCreateTensorDescriptor(&fcl->d_after_bn_desc));
  chkCUDNN(cudnnCreateTensorDescriptor(&fcl->d_after_conv_desc));
  chkCUDNN(cudnnCreateTensorDescriptor(&fcl->d_before_softmax_desc));
  chkCUDNN(cudnnCreateTensorDescriptor(&fcl->d_after_softmax_desc));

  chkCUDNN(cudnnCreateConvolutionDescriptor(&fcl->conv_desc));
  chkCUDNN(cudnnCreateFilterDescriptor(&fcl->filter_desc));
  chkCUDNN(cudnnCreateTensorDescriptor(&fcl->bn_desc));

  chkCUDNN(cudnnCreateFilterDescriptor(&fcl->d_filter_desc));
  chkCUDNN(cudnnCreateTensorDescriptor(&fcl->d_bn_desc));

  chkCUDNN(cudnnSetFilter4dDescriptor(fcl->filter_desc,
    CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
    out_features, 1, 1, in_features));
  chkCUDNN(cudnnSetFilter4dDescriptor(fcl->d_filter_desc,
    CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
    out_features, 1, 1, in_features));

  chkCUDNN(cudnnSetConvolution2dDescriptor(fcl->conv_desc,
    0, 0, 1, 1, 1, 1, CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT));

  MALLOC_TENSOR_FLOATZ(&fcl->bn_result_running_mean, 1, in_features, 1, 1);
  MALLOC_TENSOR_FLOATZ(&fcl->bn_result_running_var, 1, in_features, 1, 1);
  float *tmp = malloc(sizeof(float) * fcl->in_features);
  for (int i = 0; i < fcl->in_features; i++) {
    tmp[i] = 1.0f;
  }
  chkCUDA(cudaMemcpy(fcl->bn_result_running_var, tmp, sizeof(float) * fcl->in_features, cudaMemcpyHostToDevice));
  free(tmp);

  fcl->bn_scale = fcl->bn_bias = fcl->filter = NULL;
  fcl->d_bn_scale = fcl->d_bn_bias = fcl->d_filter = NULL;

  fcl->off_d_bn_scale = *off_d;
  *off_d += in_features;
  fcl->off_d_bn_bias = *off_d;
  *off_d += in_features;
  fcl->off_d_filter = *off_d;
  *off_d += out_features * in_features;

  if (filter != NULL) {
    chkCUDA(cudaMemcpy(fcl->filter, filter,
      sizeof(float) * out_features * in_features, cudaMemcpyHostToDevice));
  }

  if (bn_scale != NULL && bn_bias != NULL) {
    chkCUDA(cudaMemcpy(fcl->bn_scale, bn_scale,
      sizeof(float) * fcl->in_features, cudaMemcpyHostToDevice));
    chkCUDA(cudaMemcpy(fcl->bn_bias, bn_bias,
      sizeof(float) * fcl->in_features, cudaMemcpyHostToDevice));
  }

  fcl->buf_d_filter = NULL;
  fcl->buf_d_bn_scale = NULL;
  fcl->buf_d_bn_bias = NULL;

  MALLOC_TENSOR_FLOAT(&fcl->after_bn, max_batch, 1, 1, fcl->in_features);
  MALLOC_TENSOR_FLOAT(&fcl->after_conv, max_batch, 1, 1, fcl->out_features);
  MALLOC_TENSOR_FLOAT(&fcl->after_softmax, max_batch, 1, 1, fcl->out_features);
  MALLOC_TENSOR_FLOAT(&fcl->d_after_conv, max_batch, 1, 1, fcl->out_features);
  MALLOC_TENSOR_FLOAT(&fcl->d_after_bn, max_batch, 1, 1, fcl->in_features);
  MALLOC_TENSOR_FLOAT(&fcl->d_input, max_batch, 1, 1, fcl->in_features);
}

void set_fc_layer(fc_layer *fcl, float *input, float **output,
  float *d_output, float **d_input, int batch_size, bool is_training)
{
  if (fcl->bn_scale == NULL) {
    fcl->d_bn_scale = d + fcl->off_d_bn_scale;
    fcl->d_bn_bias = d + fcl->off_d_bn_bias;
    fcl->d_filter = d + fcl->off_d_filter;

    fcl->bn_scale = p + fcl->off_d_bn_scale;
    fcl->bn_bias = p + fcl->off_d_bn_bias;
    fcl->filter = p + fcl->off_d_filter;

    float k = sqrt(1.0 / fcl->in_features);
    INITIALIZE_TENSOR_URAND(fcl->bn_scale, 0, 1, fcl->in_features);
    INITIALIZE_TENSOR_URAND(fcl->filter, -k, k,
      fcl->out_features * fcl->in_features);
  }

  fcl->is_training = is_training;
  fcl->input = input;
  fcl->d_after_softmax = d_output;
  fcl->batch_size = batch_size;

  chkCUDNN(cudnnSetTensor4dDescriptor(fcl->bn_input_desc,
    CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, batch_size, fcl->in_features, 1, 1));
  chkCUDNN(cudnnSetTensor4dDescriptor(fcl->bn_output_desc,
    CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, batch_size, fcl->in_features, 1, 1));

  chkCUDNN(cudnnSetTensor4dDescriptor(fcl->input_desc,
    CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, batch_size, 1, 1, fcl->in_features));
  chkCUDNN(cudnnSetTensor4dDescriptor(fcl->after_bn_desc,
    CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, batch_size, 1, 1, fcl->in_features));
  chkCUDNN(cudnnSetTensor4dDescriptor(fcl->after_conv_desc,
    CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, batch_size, fcl->out_features, 1, 1));
  chkCUDNN(cudnnSetTensor4dDescriptor(fcl->before_softmax_desc,
    CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, batch_size, 1, 1, fcl->out_features));
  chkCUDNN(cudnnSetTensor4dDescriptor(fcl->after_softmax_desc,
    CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, batch_size, 1, 1, fcl->out_features));

  chkCUDNN(cudnnSetTensor4dDescriptor(fcl->d_bn_input_desc,
    CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, batch_size, fcl->in_features, 1, 1));
  chkCUDNN(cudnnSetTensor4dDescriptor(fcl->d_bn_output_desc,
    CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, batch_size, fcl->in_features, 1, 1));

  chkCUDNN(cudnnSetTensor4dDescriptor(fcl->d_input_desc,
    CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, batch_size, 1, 1, fcl->in_features));
  chkCUDNN(cudnnSetTensor4dDescriptor(fcl->d_after_bn_desc,
    CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, batch_size, 1, 1, fcl->in_features));
  chkCUDNN(cudnnSetTensor4dDescriptor(fcl->d_after_conv_desc,
    CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, batch_size, fcl->out_features, 1, 1));
  chkCUDNN(cudnnSetTensor4dDescriptor(fcl->d_before_softmax_desc,
    CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, batch_size, 1, 1, fcl->out_features));
  chkCUDNN(cudnnSetTensor4dDescriptor(fcl->d_after_softmax_desc,
    CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, batch_size, 1, 1, fcl->out_features));

  chkCUDNN(cudnnGetConvolutionForwardAlgorithm(fcl->cudnn,
    fcl->after_bn_desc, fcl->filter_desc, fcl->conv_desc, fcl->after_conv_desc,
    CUDNN_CONVOLUTION_FWD_PREFER_FASTEST, 0, &fcl->fwd_algo));
  chkCUDNN(cudnnGetConvolutionForwardWorkspaceSize(fcl->cudnn,
    fcl->after_bn_desc, fcl->filter_desc, fcl->conv_desc, fcl->after_conv_desc,
    fcl->fwd_algo, &fcl->workspace_bytes));

  chkCUDNN(cudnnGetConvolutionBackwardDataAlgorithm(fcl->cudnn,
    fcl->filter_desc, fcl->d_after_conv_desc, fcl->conv_desc,
    fcl->d_after_bn_desc, CUDNN_CONVOLUTION_BWD_DATA_PREFER_FASTEST,
    0, &fcl->bwd_data_algo));
  chkCUDNN(cudnnGetConvolutionBackwardDataWorkspaceSize(fcl->cudnn,
    fcl->filter_desc, fcl->d_after_conv_desc, fcl->conv_desc,
    fcl->d_after_bn_desc, fcl->bwd_data_algo, &fcl->workspace_bwd_data_bytes));

  chkCUDNN(cudnnGetConvolutionBackwardFilterAlgorithm(fcl->cudnn,
    fcl->after_bn_desc, fcl->d_after_conv_desc, fcl->conv_desc,
    fcl->d_filter_desc, CUDNN_CONVOLUTION_BWD_FILTER_PREFER_FASTEST,
    0, &fcl->bwd_filter_algo));
  chkCUDNN(cudnnGetConvolutionBackwardFilterWorkspaceSize(fcl->cudnn,
    fcl->after_bn_desc, fcl->d_after_conv_desc, fcl->conv_desc,
    fcl->d_filter_desc, fcl->bwd_filter_algo, &fcl->workspace_bwd_filter_bytes));

  fcl->workspace = get_global_workspace(fcl->workspace_bytes);
  fcl->workspace_bwd_data = get_global_workspace(fcl->workspace_bwd_data_bytes);
  fcl->workspace_bwd_filter = get_global_workspace(fcl->workspace_bwd_filter_bytes);

  chkCUDNN(cudnnDeriveBNTensorDescriptor(fcl->bn_desc, fcl->bn_input_desc,
    CUDNN_BATCHNORM_PER_ACTIVATION));
  chkCUDNN(cudnnDeriveBNTensorDescriptor(fcl->d_bn_desc, fcl->d_bn_input_desc,
    CUDNN_BATCHNORM_PER_ACTIVATION));

  *d_input = fcl->d_input;
  *output = fcl->after_softmax;
} 

void train_fwd_fc_layer(fc_layer *fcl)
{
  float one = 1.0;
  float zero = 0.0;

  if (fcl->is_training == true) {
    chkCUDNN(cudnnBatchNormalizationForwardTraining(fcl->cudnn,
      CUDNN_BATCHNORM_PER_ACTIVATION, &one, &zero, fcl->bn_input_desc, fcl->input,
      fcl->bn_output_desc, fcl->after_bn, fcl->bn_desc,
      fcl->bn_scale, fcl->bn_bias, 0.1,
      fcl->bn_result_running_mean, fcl->bn_result_running_var,
      1e-05, NULL, NULL));
  } else {
    chkCUDNN(cudnnBatchNormalizationForwardInference(fcl->cudnn,
      CUDNN_BATCHNORM_PER_ACTIVATION, &one, &zero, fcl->bn_input_desc, fcl->input,
      fcl->bn_output_desc, fcl->after_bn, fcl->bn_desc,
      fcl->bn_scale, fcl->bn_bias,
      fcl->bn_result_running_mean, fcl->bn_result_running_var,
      1e-05));
  }

  float *conv_output = (fcl->is_training ? fcl->after_softmax : fcl->after_conv);

  chkCUDNN(cudnnConvolutionForward(fcl->cudnn, &one,
    fcl->after_bn_desc, fcl->after_bn, fcl->filter_desc, fcl->filter,
    fcl->conv_desc, fcl->fwd_algo, *fcl->workspace, fcl->workspace_bytes,
    &zero, fcl->after_conv_desc, conv_output));

  if (fcl->is_training == false) {
    chkCUDNN(cudnnSoftmaxForward(fcl->cudnn, CUDNN_SOFTMAX_ACCURATE,
      CUDNN_SOFTMAX_MODE_INSTANCE, &one, fcl->before_softmax_desc, fcl->after_conv,
      &zero, fcl->after_softmax_desc, fcl->after_softmax));
  }
}

void train_bwd_fc_layer(fc_layer *fcl)
{
  float one = 1.0;
  float zero = 0.0;

  chkCUDNN(cudnnConvolutionBackwardData(fcl->cudnn,
    &one, fcl->filter_desc, fcl->filter,
    fcl->d_after_conv_desc, fcl->d_after_softmax, fcl->conv_desc,
    fcl->bwd_data_algo, *fcl->workspace_bwd_data, fcl->workspace_bwd_data_bytes,
    &zero, fcl->d_after_bn_desc, fcl->d_after_bn));
  chkCUDNN(cudnnConvolutionBackwardFilter(fcl->cudnn,
    &one, fcl->after_bn_desc, fcl->after_bn,
    fcl->d_after_conv_desc, fcl->d_after_softmax, fcl->conv_desc,
    fcl->bwd_filter_algo, *fcl->workspace_bwd_filter,
    fcl->workspace_bwd_filter_bytes, &zero, fcl->d_filter_desc, fcl->d_filter));

  chkCUDNN(cudnnBatchNormalizationBackward(fcl->cudnn,
    CUDNN_BATCHNORM_PER_ACTIVATION, &one, &zero, &one, &zero,
    fcl->bn_input_desc, fcl->input, fcl->d_bn_output_desc, fcl->d_after_bn,
    fcl->d_bn_input_desc, fcl->d_input, fcl->d_bn_desc, fcl->bn_scale, fcl->d_bn_scale,
    fcl->d_bn_bias, 1e-05, NULL, NULL));
}

void free_fc_layer(fc_layer *fcl)
{
}

