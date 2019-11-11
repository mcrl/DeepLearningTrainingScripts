#ifndef _RCNT_H_
#define _RCNT_H_

#include <stdbool.h>

#include <cudnn.h>
#include <cuda.h>

typedef struct rcnt_layer_s {
  cudnnHandle_t cudnn;

  bool is_training;
  float zero;

  int batch_size;
  int input_size;
  int hidden_size;
  int seq_length;
  int *cnt_seqs;

  int dim_weight[3];
  int dim_input[3];
  int dim_output[3];
  int dim_hidden_state[3];

  int stride_input[3];
  int stride_output[3];
  int stride_hidden_state[3];

  cudnnTensorDescriptor_t x;
  cudnnTensorDescriptor_t before_bn_desc;
  cudnnTensorDescriptor_t after_bn_desc;
  cudnnRNNDataDescriptor_t input_rcnt_desc;
  cudnnRNNDataDescriptor_t after_rcnt_desc;
  cudnnTensorDescriptor_t *_input_rcnt_desc;

  cudnnTensorDescriptor_t d_before_bn_desc;
  cudnnTensorDescriptor_t d_after_bn_desc;
  cudnnRNNDataDescriptor_t d_input_rcnt_desc;
  cudnnRNNDataDescriptor_t d_after_rcnt_desc;

  cudnnTensorDescriptor_t a_desc;
  cudnnTensorDescriptor_t c_desc;

  cudnnRNNDescriptor_t rcnt_desc;
  cudnnReduceTensorDescriptor_t reduce_desc;
  cudnnTensorDescriptor_t bn_desc;
  cudnnTensorDescriptor_t hs_output_desc;
  cudnnFilterDescriptor_t weight_desc;

  cudnnTensorDescriptor_t d_bn_desc;
  cudnnTensorDescriptor_t d_hs_output_desc;
  cudnnFilterDescriptor_t d_weight_desc;

  cudnnRNNMode_t mode;

  float *before_bn;
  float *input;
  float *packed_seq;
  float *after_rcnt;
  float *padded_seq;
  float *summed_seq;
  float *output;

  float *d_before_bn;
  float *d_input;
  float *d_packed_seq;
  float *d_after_rcnt;
  float *d_padded_seq;
  float *d_summed_seq;

  float *hs_output;
  float *weight;

  float *d_hs_output;
  float *d_weight;

  int off_d_weight;

  void **reduce_workspace;
  size_t reduce_workspace_bytes;

  void **workspace;
  size_t workspace_bytes;

  void *reservespace;
  size_t reservespace_bytes;

  bool bn_first;
  float *bn_scale;
  float *bn_bias;
  float *bn_result_running_mean;
  float *bn_result_running_var;

  float *d_bn_scale;
  float *d_bn_bias;

  int off_d_bn_bias;
  int off_d_bn_scale;

  float *buf_d_bn_scale;
  float *buf_d_bn_bias;
  float *buf_d_weight;

} rcnt_layer;

void init_rcnt_layer(rcnt_layer *rcl, cudnnHandle_t cudnn, cudnnRNNMode_t mode,
  int input_size, int hidden_size, int max_batch_size, int max_seq,
  bool bn_first, float *weights, float *biases, float *bn_scale, float *bn_bias,
  int *off_d);

void set_rcnt_layer(rcnt_layer *rcl, float *input, float **output,
  float *d_output, float **d_input, int batch_size, int seq_length,
  int *cnt_seqs, int *seqs, bool is_training);
void set_rcnt_layer_params(rcnt_layer *rcl, float *weights, float *biases);
float* get_rcnt_matrix_ptr(rcnt_layer *rcl, int pl, int l);
float* get_rcnt_d_matrix_ptr(rcnt_layer *rcl, int pl, int l);
float* get_rcnt_d_bias_ptr(rcnt_layer *rcl, int pl, int l);

void train_fwd_rcnt_layer(rcnt_layer *rcl);
void train_bwd_rcnt_layer(rcnt_layer *rcl);
void update_rcnt_layer(rcnt_layer *rcl, float lr, float clip);

void free_rcnt_layer(rcnt_layer *rcl);

#endif

