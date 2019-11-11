#ifndef _FC_H_
#define _FC_H_

#include <stdbool.h>

#include <cudnn.h>
#include <cuda.h>

typedef struct fc_layer_s {
  cudnnHandle_t cudnn;
 
  bool is_training;

  int batch_size;
  int in_features;
  int out_features;

  cudnnTensorDescriptor_t bn_input_desc;
  cudnnTensorDescriptor_t bn_output_desc;
  cudnnTensorDescriptor_t input_desc;
  cudnnTensorDescriptor_t after_bn_desc;
  cudnnTensorDescriptor_t after_conv_desc;
  cudnnTensorDescriptor_t before_softmax_desc;
  cudnnTensorDescriptor_t after_softmax_desc;
 
  cudnnTensorDescriptor_t d_bn_input_desc;
  cudnnTensorDescriptor_t d_bn_output_desc;
  cudnnTensorDescriptor_t d_input_desc;
  cudnnTensorDescriptor_t d_after_bn_desc;
  cudnnTensorDescriptor_t d_after_conv_desc;
  cudnnTensorDescriptor_t d_before_softmax_desc;
  cudnnTensorDescriptor_t d_after_softmax_desc;

  cudnnConvolutionDescriptor_t conv_desc;
  cudnnFilterDescriptor_t filter_desc;
  cudnnConvolutionFwdAlgo_t fwd_algo;
  cudnnTensorDescriptor_t bn_desc;
  cudnnTensorDescriptor_t d_bn_desc;

  cudnnFilterDescriptor_t d_filter_desc;
  cudnnConvolutionBwdDataAlgo_t bwd_data_algo;
  cudnnConvolutionBwdFilterAlgo_t bwd_filter_algo;

  float *input;
  float *after_bn;
  float *after_conv;
  float *after_softmax;

  float *d_after_softmax;
  float *d_after_conv;
  float *d_after_bn;
  float *d_input;

  float *filter;
  float *bn_scale;
  float *bn_bias;
  float *bn_result_running_mean;
  float *bn_result_running_var;

  float *d_filter;
  float *d_bn_scale;
  float *d_bn_bias;

  int off_d_filter;
  int off_d_bn_scale;
  int off_d_bn_bias;

  float *buf_d_filter;
  float *buf_d_bn_scale;
  float *buf_d_bn_bias;

  void **workspace;
  size_t workspace_bytes;

  void **workspace_bwd_data;
  size_t workspace_bwd_data_bytes;

  void **workspace_bwd_filter;
  size_t workspace_bwd_filter_bytes;

} fc_layer;

void init_fc_layer(fc_layer *fcl, cudnnHandle_t cudnn,
  int in_features, int out_features,
  float *filter, float *bn_scale, float *bn_bias, int *off_d, int max_batch);

void set_fc_layer(fc_layer *fcl, float *input, float **output,
  float *d_output, float **d_input, int batch_size, bool is_training);

void train_fwd_fc_layer(fc_layer *fcl);
void train_bwd_fc_layer(fc_layer *fcl);
void update_fc_layer(fc_layer *fcl, float lr, float clip);

void free_fc_layer(fc_layer *fcl);

#endif

