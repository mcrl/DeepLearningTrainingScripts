#include <math.h>
#include <time.h>

#include <builtin_types.h>
#include <cuda_runtime_api.h>
#include <cuda.h>
#include <cudnn.h>

#include "deepspeech.h"
#include "deepspeech_cuda.h"
#include "conv.h"
#include "params.h"
#include "utils.h"

void init_conv_layer(conv_layer *cvl, cudnnHandle_t cudnn,
  int filter_height, int filter_width, int pad_height, int pad_width,
  int stride_x, int stride_y, int out, int in,
  float *filter, float *bn_scale, float *bn_bias, int *off_d,
  int max_batch, int max_height, int max_width)
{
  cvl->cudnn = cudnn;
  cvl->filter_height = filter_height;
  cvl->filter_width = filter_width;
  cvl->pad_height = pad_height;
  cvl->pad_width = pad_width;
  cvl->stride_x = stride_x;
  cvl->stride_y = stride_y;
  cvl->channels_out = out;
  cvl->channels_in = in;

  chkCUDNN(cudnnCreateTensorDescriptor(&cvl->input_desc));
  chkCUDNN(cudnnCreateTensorDescriptor(&cvl->after_conv_desc));
  chkCUDNN(cudnnCreateTensorDescriptor(&cvl->after_bn_desc));
  chkCUDNN(cudnnCreateTensorDescriptor(&cvl->after_act_desc));

  chkCUDNN(cudnnCreateTensorDescriptor(&cvl->d_input_desc));
  chkCUDNN(cudnnCreateTensorDescriptor(&cvl->d_after_conv_desc));
  chkCUDNN(cudnnCreateTensorDescriptor(&cvl->d_after_bn_desc));
  chkCUDNN(cudnnCreateTensorDescriptor(&cvl->d_after_act_desc));

  chkCUDNN(cudnnCreateConvolutionDescriptor(&cvl->conv_desc));
  chkCUDNN(cudnnCreateFilterDescriptor(&cvl->filter_desc));
  chkCUDNN(cudnnCreateTensorDescriptor(&cvl->bn_desc));
  chkCUDNN(cudnnCreateActivationDescriptor(&cvl->act_desc));

  chkCUDNN(cudnnCreateFilterDescriptor(&cvl->d_filter_desc));
  chkCUDNN(cudnnCreateTensorDescriptor(&cvl->d_bn_desc));

  chkCUDNN(cudnnSetFilter4dDescriptor(cvl->filter_desc,
    CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
    cvl->channels_out, cvl->channels_in,
    cvl->filter_height, cvl->filter_width));
  chkCUDNN(cudnnSetConvolution2dDescriptor(cvl->conv_desc,
    cvl->pad_height, cvl->pad_width, cvl->stride_x, cvl->stride_y, 1, 1,
    CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT));
  chkCUDNN(cudnnSetActivationDescriptor(cvl->act_desc,
    CUDNN_ACTIVATION_CLIPPED_RELU, CUDNN_NOT_PROPAGATE_NAN, 20.0));
  
  chkCUDNN(cudnnSetFilter4dDescriptor(cvl->d_filter_desc,
    CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
    cvl->channels_out, cvl->channels_in,
    cvl->filter_height, cvl->filter_width));

  MALLOC_TENSOR_FLOATZ(&cvl->bn_result_running_mean, 1, out, 1, 1);
  MALLOC_TENSOR_FLOATZ(&cvl->bn_result_running_var, 1, out, 1, 1);
  MALLOC_TENSOR_FLOATZ(&cvl->bn_result_save_mean, 1, out, 1, 1);
  MALLOC_TENSOR_FLOATZ(&cvl->bn_result_save_var, 1, out, 1, 1);

  cvl->filter = cvl->bn_scale = cvl->bn_bias = NULL;
  cvl->d_filter = cvl->d_bn_scale = cvl->d_bn_bias = NULL;

  cvl->off_d_filter = *off_d;
  *off_d += out * in * filter_height * filter_width;
  cvl->off_d_bn_scale = *off_d;
  *off_d += out;
  cvl->off_d_bn_bias = *off_d;
  *off_d += out;

  float *tmp = (float *)malloc(sizeof(float) * out);
  for (int i = 0; i < 32; i++) {
    tmp[i] = 1.0f;
  }
  chkCUDA(cudaMemcpy(cvl->bn_result_running_var, tmp, sizeof(float) * out, cudaMemcpyHostToDevice));
  free(tmp);

  if (filter != NULL) {
    chkCUDA(cudaMemcpy(cvl->filter, filter,
      sizeof(float) * cvl->channels_out * cvl->channels_in *
      cvl->filter_height * cvl->filter_width,
      cudaMemcpyHostToDevice));
  }
  if (bn_scale != NULL) {
    chkCUDA(cudaMemcpy(cvl->bn_scale, bn_scale,
      sizeof(float) * cvl->channels_out, cudaMemcpyHostToDevice));
  }
  if (bn_bias != NULL) {
    chkCUDA(cudaMemcpy(cvl->bn_bias, bn_bias,
      sizeof(float) * cvl->channels_out, cudaMemcpyHostToDevice));
  }

  cvl->buf_d_filter = NULL;
  cvl->buf_d_bn_scale = NULL;
  cvl->buf_d_bn_bias = NULL;

  const int height_after_conv = CALC_SIZE(max_height, cvl->filter_height,
    cvl->pad_height, cvl->stride_x);
  const int width_after_conv = CALC_SIZE(max_width, cvl->filter_width,
    cvl->pad_width, cvl->stride_y);
  const int height_after_bnact = height_after_conv;
  const int width_after_bnact = width_after_conv;

  MALLOC_TENSOR_FLOATZ(&cvl->after_conv, max_batch, cvl->channels_out,
    height_after_conv, width_after_conv);
  MALLOC_TENSOR_FLOAT(&cvl->after_bn, max_batch, cvl->channels_out,
    height_after_bnact, width_after_bnact);
  MALLOC_TENSOR_FLOAT(&cvl->after_act, max_batch, cvl->channels_out,
    height_after_bnact, width_after_bnact);

  MALLOC_TENSOR_FLOAT(&cvl->d_input, max_batch, cvl->channels_in,
    max_height, max_width);
  MALLOC_TENSOR_FLOAT(&cvl->d_after_conv, max_batch, cvl->channels_out,
    height_after_conv, width_after_conv);
  MALLOC_TENSOR_FLOAT(&cvl->d_after_bn, max_batch, cvl->channels_out,
    height_after_bnact, width_after_bnact);
}

void set_conv_layer(conv_layer *cvl, float *input, float **output,
  float *d_output, float **d_input, int batch_size, int input_height,
  int input_width, bool is_training)
{
  if (cvl->filter == NULL) {
    cvl->d_filter = d + cvl->off_d_filter;
    cvl->d_bn_bias = d + cvl->off_d_bn_bias;
    cvl->d_bn_scale = d + cvl->off_d_bn_scale;

    cvl->filter = p + cvl->off_d_filter;
    cvl->bn_bias = p + cvl->off_d_bn_bias;
    cvl->bn_scale = p + cvl->off_d_bn_scale;

    float k = sqrt(1.0 / (cvl->channels_in * cvl->filter_height *
      cvl->filter_width));
    INITIALIZE_TENSOR_URAND(cvl->bn_scale, 0, 1, cvl->channels_out);
    INITIALIZE_TENSOR_URAND(cvl->filter, -k, k,
      cvl->channels_out * cvl->channels_in * cvl->filter_height *
      cvl->filter_width);
  }

  cvl->is_training = is_training;

  const int height_after_conv = CALC_SIZE(input_height, cvl->filter_height,
    cvl->pad_height, cvl->stride_x);
  const int width_after_conv = CALC_SIZE(input_width, cvl->filter_width,
    cvl->pad_width, cvl->stride_y);
  const int height_after_bnact = height_after_conv;
  const int width_after_bnact = width_after_conv;

  chkCUDNN(cudnnSetTensor4dDescriptor(cvl->input_desc,
    CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, batch_size, cvl->channels_in,
    input_height, input_width));
  chkCUDNN(cudnnSetTensor4dDescriptor(cvl->after_conv_desc,
    CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, batch_size, cvl->channels_out,
    height_after_conv, width_after_conv));
  chkCUDNN(cudnnSetTensor4dDescriptor(cvl->after_bn_desc,
    CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, batch_size, cvl->channels_out,
    height_after_bnact, width_after_bnact));
  chkCUDNN(cudnnSetTensor4dDescriptor(cvl->after_act_desc,
    CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, batch_size, cvl->channels_out,
    height_after_bnact, width_after_bnact));

  chkCUDNN(cudnnSetTensor4dDescriptor(cvl->d_input_desc,
    CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, batch_size, cvl->channels_in,
    input_height, input_width));
  chkCUDNN(cudnnSetTensor4dDescriptor(cvl->d_after_conv_desc,
    CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, batch_size, cvl->channels_out,
    height_after_conv, width_after_conv));
  chkCUDNN(cudnnSetTensor4dDescriptor(cvl->d_after_bn_desc,
    CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, batch_size, cvl->channels_out,
    height_after_bnact, width_after_bnact));
  chkCUDNN(cudnnSetTensor4dDescriptor(cvl->d_after_act_desc,
    CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, batch_size, cvl->channels_out,
    height_after_bnact, width_after_bnact));

  chkCUDNN(cudnnGetConvolutionForwardAlgorithm(cvl->cudnn,
    cvl->input_desc, cvl->filter_desc, cvl->conv_desc, cvl->after_conv_desc,
    CUDNN_CONVOLUTION_FWD_PREFER_FASTEST, 0, &cvl->fwd_algo));
  chkCUDNN(cudnnGetConvolutionForwardWorkspaceSize(cvl->cudnn,
    cvl->input_desc, cvl->filter_desc, cvl->conv_desc, cvl->after_conv_desc,
    cvl->fwd_algo, &cvl->conv_workspace_bytes));
  cvl->conv_workspace = get_global_workspace(cvl->conv_workspace_bytes);

  chkCUDNN(cudnnGetConvolutionBackwardDataAlgorithm(cvl->cudnn,
    cvl->filter_desc, cvl->d_after_conv_desc, cvl->conv_desc, cvl->d_input_desc,
    CUDNN_CONVOLUTION_BWD_DATA_PREFER_FASTEST, 0, &cvl->bwd_data_algo));
  chkCUDNN(cudnnGetConvolutionBackwardDataWorkspaceSize(cvl->cudnn,
    cvl->filter_desc, cvl->d_after_conv_desc, cvl->conv_desc, cvl->d_input_desc,
    cvl->bwd_data_algo, &cvl->conv_workspace_bwd_data_bytes));
  cvl->conv_workspace_bwd_data =
    get_global_workspace(cvl->conv_workspace_bwd_data_bytes);

  chkCUDNN(cudnnGetConvolutionBackwardFilterAlgorithm(cvl->cudnn,
    cvl->input_desc, cvl->d_after_conv_desc, cvl->conv_desc, cvl->d_filter_desc,
    CUDNN_CONVOLUTION_BWD_FILTER_PREFER_FASTEST, 0, &cvl->bwd_filter_algo));
  cvl->bwd_filter_algo = CUDNN_CONVOLUTION_BWD_FILTER_ALGO_1;
  chkCUDNN(cudnnGetConvolutionBackwardFilterWorkspaceSize(cvl->cudnn,
    cvl->input_desc, cvl->d_after_conv_desc, cvl->conv_desc, cvl->d_filter_desc,
    cvl->bwd_filter_algo, &cvl->conv_workspace_bwd_filter_bytes));
  cvl->conv_workspace_bwd_filter =
    get_global_workspace(cvl->conv_workspace_bwd_filter_bytes);

  chkCUDNN(cudnnDeriveBNTensorDescriptor(cvl->bn_desc, cvl->after_conv_desc,
    CUDNN_BATCHNORM_SPATIAL));
  chkCUDNN(cudnnDeriveBNTensorDescriptor(cvl->d_bn_desc, cvl->d_after_act_desc,
    CUDNN_BATCHNORM_SPATIAL));

  *output = cvl->after_act;
  cvl->input = input; 
  cvl->output_height = height_after_bnact;
  cvl->output_width = width_after_bnact;

  cvl->d_after_act = d_output;
  *d_input = cvl->d_input;
}

void train_fwd_conv_layer(conv_layer *cvl)
{
  float one = 1.0;
  float zero = 0.0;

  chkCUDNN(cudnnConvolutionForward(cvl->cudnn, &one,
    cvl->input_desc, cvl->input, cvl->filter_desc, cvl->filter, cvl->conv_desc,
    cvl->fwd_algo, *cvl->conv_workspace, cvl->conv_workspace_bytes,
    &zero, cvl->after_conv_desc, cvl->after_conv));

  if (cvl->is_training == true) {
    chkCUDNN(cudnnBatchNormalizationForwardTraining(cvl->cudnn,
      CUDNN_BATCHNORM_SPATIAL, &one, &zero,
      cvl->after_conv_desc, cvl->after_conv,
      cvl->after_bn_desc, cvl->after_bn, cvl->bn_desc,
      cvl->bn_scale, cvl->bn_bias, 0.1,
      cvl->bn_result_running_mean, cvl->bn_result_running_var,
      1e-05, cvl->bn_result_save_mean, cvl->bn_result_save_var));
  } else {
    chkCUDNN(cudnnBatchNormalizationForwardInference(cvl->cudnn,
      CUDNN_BATCHNORM_SPATIAL, &one, &zero,
      cvl->after_conv_desc, cvl->after_conv,
      cvl->after_bn_desc, cvl->after_bn, cvl->bn_desc,
      cvl->bn_scale, cvl->bn_bias,
      cvl->bn_result_running_mean, cvl->bn_result_running_var,
      1e-05));
  }

  chkCUDNN(cudnnActivationForward(cvl->cudnn, cvl->act_desc,
    &one, cvl->after_bn_desc, cvl->after_bn, &zero,
    cvl->after_act_desc, cvl->after_act));
}

void train_bwd_conv_layer(conv_layer *cvl)
{
  assert(cvl->d_after_act != NULL);

  float one = 1.0;
  float zero = 0.0;

  START_STOPWATCH {
    chkCUDNN(cudnnActivationBackward(cvl->cudnn, cvl->act_desc,
      &one, cvl->after_act_desc, cvl->after_act,
      cvl->d_after_act_desc, cvl->d_after_act, cvl->after_bn_desc, cvl->after_bn,
      &zero, cvl->d_after_bn_desc, cvl->d_after_bn));

    chkCUDNN(cudnnBatchNormalizationBackward(cvl->cudnn,
      CUDNN_BATCHNORM_SPATIAL, &one, &zero, &one, &zero,
      cvl->after_conv_desc, cvl->after_conv,
      cvl->d_after_bn_desc, cvl->d_after_bn,
      cvl->d_after_conv_desc, cvl->d_after_conv,
      cvl->d_bn_desc, cvl->bn_scale, cvl->d_bn_scale, cvl->d_bn_bias,
      1e-05, cvl->bn_result_save_mean, cvl->bn_result_save_var));

    chkCUDNN(cudnnConvolutionBackwardData(cvl->cudnn,
      &one, cvl->filter_desc, cvl->filter,
      cvl->d_after_conv_desc, cvl->d_after_conv, cvl->conv_desc,
      cvl->bwd_data_algo,
      *cvl->conv_workspace_bwd_data, cvl->conv_workspace_bwd_data_bytes,
      &zero, cvl->d_input_desc, cvl->d_input));

    chkCUDNN(cudnnConvolutionBackwardFilter(cvl->cudnn,
      &one, cvl->input_desc, cvl->input,
      cvl->d_after_conv_desc, cvl->d_after_conv, cvl->conv_desc,
      cvl->bwd_filter_algo,
      *cvl->conv_workspace_bwd_filter, cvl->conv_workspace_bwd_filter_bytes,
      &zero, cvl->d_filter_desc, cvl->d_filter));
  } STOP_STOPWATCH("conv cudnn func");
}

void free_conv_layer(conv_layer *cvl)
{
}

