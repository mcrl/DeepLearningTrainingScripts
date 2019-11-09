#ifndef _CONV_H_
#define _CONV_H_

#include <stdbool.h>

#include <cudnn.h>
#include <cuda.h>

typedef struct conv_layer_s {
  cudnnHandle_t cudnn;

  bool is_training;

  int filter_height, filter_width;
  int pad_height, pad_width;
  int stride_x, stride_y;
  int channels_out, channels_in;
  int output_height, output_width;

  cudnnTensorDescriptor_t input_desc;
  cudnnTensorDescriptor_t after_conv_desc;
  cudnnTensorDescriptor_t after_bn_desc;
  cudnnTensorDescriptor_t after_act_desc; 

  cudnnTensorDescriptor_t d_input_desc;
  cudnnTensorDescriptor_t d_after_conv_desc;
  cudnnTensorDescriptor_t d_after_bn_desc;
  cudnnTensorDescriptor_t d_after_act_desc; 

  cudnnConvolutionDescriptor_t conv_desc;
  cudnnFilterDescriptor_t filter_desc;
  cudnnConvolutionFwdAlgo_t fwd_algo;
  cudnnTensorDescriptor_t bn_desc;
  cudnnActivationDescriptor_t act_desc;

  cudnnFilterDescriptor_t d_filter_desc;
  cudnnConvolutionBwdDataAlgo_t bwd_data_algo;
  cudnnConvolutionBwdFilterAlgo_t bwd_filter_algo;
  cudnnTensorDescriptor_t d_bn_desc;

  float *input;
  float *after_conv;
  float *after_bn;
  float *after_act;

  float *d_input;
  float *d_after_conv;
  float *d_after_bn;
  float *d_after_act;

  float *filter;
  float *bn_scale;
  float *bn_bias;
  float *bn_result_running_mean;
  float *bn_result_running_var;

  float *bn_result_save_mean;
  float *bn_result_save_var;

  float *d_filter;
  float *d_bn_scale;
  float *d_bn_bias;

  int off_d_filter;
  int off_d_bn_scale;
  int off_d_bn_bias;

  float *buf_d_filter;
  float *buf_d_bn_scale;
  float *buf_d_bn_bias;

  void **conv_workspace;
  void **conv_workspace_bwd_data;
  void **conv_workspace_bwd_filter;

  size_t conv_workspace_bytes;
  size_t conv_workspace_bwd_data_bytes;
  size_t conv_workspace_bwd_filter_bytes;

} conv_layer;

void init_conv_layer(conv_layer *cvl, cudnnHandle_t cudnn,
  int filter_height, int filter_width, int pad_height, int pad_width,
  int stride_x, int stride_y, int out, int in,
  float *filter, float *bn_scale, float *bn_bias, int *off_d,
  int max_batch, int max_height, int max_width);

void set_conv_layer(conv_layer *cvl, float *input, float **output,
  float *d_output, float **d_input, int batch_size, int input_height,
  int input_width, bool is_training);

void train_fwd_conv_layer(conv_layer *cvl);
void train_bwd_conv_layer(conv_layer *cvl);
void free_conv_layer(conv_layer *cvl);

#endif

