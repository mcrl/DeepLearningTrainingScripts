#ifndef _LAYER_H_
#define _LAYER_H_

#include <stdbool.h>

#include <cuda.h>
#include <cudnn.h>

#include "execute.h"

#define DET_CUDNN
//#define TIME_LAYER
//#define PRINT_LOSS

//VGG CONNECTION
#define CONNECT_INPUT(in) \
do {\
  alloc_buffer((in).output);\
  alloc_buffer((in).d_output);\
} while (0)

#define CONNECT(up, down) \
do {\
  share_buffer((down).input, (up).output);\
  share_buffer((down).d_input, (up).d_output);\
  alloc_buffer((down).output);\
  alloc_buffer((down).d_output);\
} while (0)

#define CONNECT_BIAS(up, bias, down) \
do {\
  share_buffer((bias).output, (up).output);\
  share_buffer((bias).d_output, (up).d_output);\
  CONNECT(bias, down);\
} while (0)

//RESNET CONNECTION
#define CONNECT_DIAMOND_RES(l_branch, l_elt, l_up, l_down) \
do {\
  (l_elt).input1 = (l_branch).input;\
  (l_elt).input2 = (l_down).output;\
  (l_up).input = (l_branch).input;\
  (l_branch).d_output[0] = (l_elt).d_output;\
  (l_branch).d_output[1] = (l_up).d_input;\
  (l_down).d_output = (l_elt).d_output;\
} while (0)

#define CONNECT_BRANCH_RES(l_branch, l_down1, l_down2) \
do {\
  (l_down1).input = (l_branch).input;\
  (l_down2).input = (l_branch).input;\
  (l_branch).d_output[0] = (l_down1).d_input;\
  (l_branch).d_output[1] = (l_down2).d_input;\
} while (0)

#define CONNECT_ELT(l_up1, l_up2, l_elt) \
do {\
  (l_elt).input1 = (l_up1).output;\
  (l_elt).input2 = (l_up2).output;\
  (l_up1).d_output = (l_elt).d_output;\
  (l_up2).d_output = (l_elt).d_output;\
} while (0)

//DENSENET CONNECTION
#define CONNECT_DIAMOND_DENSE(l_branch, l_concat, l_up, l_down) \
do {\
  (l_concat).input[0] = (l_branch).input;\
  (l_concat).input[1] = (l_down).output;\
  (l_up).input = (l_branch).input;\
  (l_branch).d_output[0] = (l_concat).d_input[0];\
  (l_branch).d_output[1] = (l_up).d_input;\
  (l_down).d_output = (l_concat).d_input[1];\
} while (0)

//INCEPTION CONNECTION
#define CONNECT_BRANCH_I(l_branch, l_down, i) \
do {\
  (l_down).input = (l_branch).input;\
  (l_branch).d_output[i] = (l_down).d_input;\
} while (0)

#define CONNECT_CONCAT_I(l_up, l_concat, i) \
do {\
  (l_concat).input[i] = (l_up).output;\
  (l_up).d_output = (l_concat).d_input[i];\
} while (0)

//PARAM FUNCTIONS
#define INIT_CONV_RES(l) \
do {\
  float n_in = (l)->filter_height * (l)->filter_width * (l)->input_channel;\
  float n_out = (l)->filter_height * (l)->filter_width * (l)->output_channel;\
  size_t csz = (l)->filter_height * (l)->filter_width * (l)->input_channel * (l)->output_channel;\
  INITIALIZE_RAND_NORM_SCALE(param, csz, sqrt(2 / (n_out)));\
  param += set_conv_filter(l, param);\
} while (0) 

#define INIT_FC(l) \
do {\
  float n_in = (l)->in;\
  float n_out = (l)->out;\
  size_t csz = n_in * n_out;\
  INITIALIZE_RAND_SCALE(param, csz, sqrt(2 / (n_in + n_out)));\
  param += set_fc_weight(l, param);\
} while (0)

#define INIT_CONV(l) \
do {\
  float n_in = (l)->filter_height * (l)->filter_width * (l)->input_channel;\
  float n_out = (l)->filter_height * (l)->filter_width * (l)->output_channel;\
  size_t csz = (l)->filter_height * (l)->filter_width * (l)->input_channel * (l)->output_channel;\
  INITIALIZE_RAND_SCALE(param, csz, sqrt(6 / (n_in + n_out)));\
  param += set_conv_filter(l, param);\
} while (0)

#define INIT_BN(l) \
do {\
  int cnt = (l)->channel;\
  INITIALIZE_CONST(param, cnt, 1.0);\
  INITIALIZE_CONST(param + cnt, cnt, 0.0);\
  param += set_bn_vars(l, param);\
} while (0) 

#define INIT_BIAS(l) \
do {\
  int cnt = (l)->channel;\
  INITIALIZE_CONST(param, cnt, 0.0);\
  param += set_bias(l, param);\
} while (0)

#define SIZE_FC(l) \
do { sum += param_size_fc(l); } while (0)

#define SIZE_CONV(l) \
do { sum += param_size_conv(l); } while (0)

#define SIZE_BN(l) \
do { sum += param_size_bn(l); } while (0)

#define SIZE_BIAS(l) \
do { sum += param_size_bias(l); } while (0)

#define LOAD_FC(l) \
do { param += set_fc_weight(l, param); } while (0)

#define LOAD_CONV(l) \
do { param += set_conv_filter(l, param); } while (0)

#define LOAD_BN(l) \
do { param += set_bn_vars(l, param); } while (0) 

#define LOAD_BIAS(l) \
do { param += set_bias(l, param); } while (0)

#define GET_FC(l) \
do { param += get_fc_weight(l, param); } while (0)

#define GET_CONV(l) \
do { param += get_conv_filter(l, param); } while (0)

#define GET_BN(l) \
do { param += get_bn_vars(l, param); } while (0)

#define GET_BIAS(l) \
do { param += get_bias(l, param); } while (0)

typedef struct fc_layer_s {
  iterator_t iterator;
  char name[256];

  int batch_size;
  int in, out;

  cudnnConvolutionDescriptor_t conv_desc;

  cudnnConvolutionFwdAlgo_t fwd_algo;
  cudnnConvolutionBwdDataAlgo_t bwd_data_algo;
  cudnnConvolutionBwdFilterAlgo_t bwd_filter_algo;

  gpu_mem input, d_input;
  gpu_mem output, d_output;
  gpu_mem filter, d_filter;

  gpu_mem ws_fwd, ws_bwd_data, ws_bwd_filter;

  float fwd_t, bwd_data_t, bwd_weight_t, bwd_update_t;
} fc_layer;

typedef struct conv_layer_s {
  iterator_t iterator;
  char name[256];

  int filter_height, filter_width;
  int pad_height, pad_width;
  int stride_height, stride_width;

  int batch_size;
  int input_channel, input_height, input_width;
  int output_channel, output_height, output_width;

  cudnnConvolutionDescriptor_t conv_desc;

  cudnnConvolutionFwdAlgo_t fwd_algo;
  cudnnConvolutionBwdDataAlgo_t bwd_data_algo;
  cudnnConvolutionBwdFilterAlgo_t bwd_filter_algo;

  gpu_mem input, d_input;
  gpu_mem output, d_output;
  gpu_mem filter, d_filter;

  gpu_mem ws_fwd, ws_bwd_data, ws_bwd_filter;

  float fwd_t, bwd_data_t, bwd_filter_t, bwd_update_t;
} conv_layer;

typedef struct bn_layer_s {
  iterator_t iterator;
  char name[256];

  int batch_size;
  int channel, height, width;

  cudnnBatchNormMode_t mode;

  double eaf, eps;

  gpu_mem input, d_input;
  gpu_mem output, d_output;
  gpu_mem scale, d_scale;
  gpu_mem bias, d_bias;
  gpu_mem running_mean, running_var;
  gpu_mem save_mean, save_var;
  
  float fwd_t, bwd_t, bwd_update_t;
} bn_layer;

typedef enum { RELU_T } act_type;

typedef struct act_layer_s {
  iterator_t iterator;
  char name[256];

  int batch_size;
  int channel, height, width;

  act_type type;

  cudnnActivationDescriptor_t act_desc;

  gpu_mem input, d_input;
  gpu_mem output, d_output;
  
  float fwd_t, bwd_t;
} act_layer;

#define MAX_IN 16
typedef struct concat_layer_s {
  iterator_t iterator;
  char name[256];

  int batch_size;
  int input_channel[MAX_IN];
  int output_channel, height, width;

  int fan_in;

  gpu_mem input[MAX_IN], d_input[MAX_IN];
  gpu_mem output, d_output;
  
  float fwd_t, bwd_t;
} concat_layer;

typedef enum { ADD_T } elt_type;

typedef struct elt_layer_s {
  iterator_t iterator;
  char name[256];

  int batch_size;
  int channel, height, width;

  elt_type type;

  cudnnOpTensorDescriptor_t op_desc;

  gpu_mem input[2];
  gpu_mem output, d_output;
  
  float fwd_t;
} elt_layer;

#define MAX_OUT 16
typedef struct branch_layer_s {
  iterator_t iterator;
  char name[256];

  int batch_size;
  int channel, height, width;

  int fan_out;

  cudnnOpTensorDescriptor_t op_desc;

  gpu_mem input, d_input;
  gpu_mem d_output[MAX_OUT];
  
  float bwd_t;
} branch_layer;

typedef struct bias_layer_s {
  iterator_t iterator;
  char name[256];

  int batch_size;
  int channel, height, width;

  gpu_mem output, d_output;
  gpu_mem bias, d_bias;
  
  float fwd_t, bwd_t, bwd_update_t;
} bias_layer;

typedef enum {
  MAX_T, AVERAGE_T
} pool_type;

typedef struct pool_layer_s {
  iterator_t iterator;
  char name[256];

  int filter_height, filter_width;
  int pad_height, pad_width;
  int stride_height, stride_width;

  int batch_size;
  int channel;
  int input_height, input_width;
  int output_height, output_width;

  pool_type type;

  cudnnPoolingDescriptor_t pool_desc;

  gpu_mem input, d_input;
  gpu_mem output, d_output;

  float fwd_t, bwd_t;
} pool_layer;

typedef struct input_layer_s {
  iterator_t iterator;
  char name[256];

  int batch_size;
  int channel, height, width;

  gpu_mem output, d_output;
} input_layer;

typedef struct softmax_layer_s {
  iterator_t iterator;
  char name[256];

  int batch_size;
  int out;

  cudnnOpTensorDescriptor_t op_desc;

  gpu_mem input, d_input;
  gpu_mem output, d_output;
  gpu_mem label;

  float fwd_t, bwd_t;
} softmax_layer;

void init_input_layer(
    input_layer *l, const char *name,
    int batch_size, int channel, int height, int width);

void init_elt_layer(
    elt_layer *l, const char *name,
    int batch_size, int channel, int height, int width, elt_type type);

void init_bias_layer(
    bias_layer *l, const char *name,
    int batch_size, int channel, int height, int width);

void init_conv_layer(
    conv_layer *l, const char *name,
    int batch_size, int filter_height, int filter_width,
    int pad_height, int pad_width, int stride_height, int stride_width,
    int input_channel, int output_channel, int input_height, int input_width);

void init_fc_layer(
    fc_layer *l, const char *name, int batch_size, int in, int out);

void init_bn_layer(
    bn_layer *l, const char *name,
    int batch_size, int channel, int height, int width, int nth);

void init_act_layer(
    act_layer *l, const char *name,
    int batch_size, int channel, int height, int width, act_type type);

void init_pool_layer(
    pool_layer *l, const char *name,
    int batch_size, int filter_height, int filter_width, 
    int pad_height, int pad_width, int stride_height, int stride_width,
    int channel, int input_height, int input_width, pool_type type);

void init_softmax_layer(
    softmax_layer *l, const char *name, int batch_size, int out);

void init_branch_layer(
    branch_layer *l, const char *name,
    int batch_size, int fan_out, int channel, int height, int width);

void init_concat_layer(
    concat_layer *l, const char *name,
    int batch_size, int fan_in, int input_channel[], int height, int width);

void train_fwd_conv_layer(conv_layer *l);
void train_fwd_fc_layer(fc_layer *l);
void train_fwd_bn_layer(bn_layer *l);
void train_fwd_act_layer(act_layer *l);
void train_fwd_pool_layer(pool_layer *l);
void train_fwd_elt_layer(elt_layer *l);
void train_fwd_softmax_layer(softmax_layer *l);
void train_fwd_branch_layer(branch_layer *l);
void train_fwd_bias_layer(bias_layer *l);
void train_fwd_concat_layer(concat_layer *l);

void train_bwd_conv_layer(conv_layer *l);
void train_bwd_fc_layer(fc_layer *l);
void train_bwd_bn_layer(bn_layer *l);
void train_bwd_bn_res_layer(bn_layer *l);
void train_bwd_act_layer(act_layer *l);
void train_bwd_pool_layer(pool_layer *l);
void train_bwd_elt_layer(elt_layer *l);
void train_bwd_softmax_layer(softmax_layer *l);
void train_bwd_branch_layer(branch_layer *l);
void train_bwd_bias_layer(bias_layer *l);
void train_bwd_concat_layer(concat_layer *l);

void set_input(input_layer *l, float *data_in);
void set_label(softmax_layer *l, int *label_in);

void print_time_conv_layer(conv_layer *l);
void print_time_fc_layer(fc_layer *l);
void print_time_bn_layer(bn_layer *l);
void print_time_act_layer(act_layer *l);
void print_time_pool_layer(pool_layer * l);
void print_time_elt_layer(elt_layer *l);
void print_time_softmax_layer(softmax_layer *l);
void print_time_branch_layer(branch_layer *l);
void print_time_bias_layer(bias_layer *l);
void print_time_concat_layer(concat_layer *l);

void clear_time_conv_layer(conv_layer *l);
void clear_time_fc_layer(fc_layer *l);
void clear_time_bn_layer(bn_layer *l);
void clear_time_act_layer(act_layer * l);
void clear_time_pool_layer(pool_layer * l);
void clear_time_elt_layer(elt_layer *l);
void clear_time_softmax_layer(softmax_layer *l);
void clear_time_branch_layer(branch_layer *l);
void clear_time_bias_layer(bias_layer *l);
void clear_time_concat_layer(concat_layer *l);

int get_conv_filter(conv_layer *l, float *filter);
int set_conv_filter(conv_layer *l, float *filter);
int get_fc_weight(fc_layer *l, float *weight);
int set_fc_weight(fc_layer *l, float *weight);
int get_bn_vars(bn_layer *l, float *bn);
int set_bn_vars(bn_layer *l, float *bn);
int get_bias(bias_layer *l, float *bias);
int set_bias(bias_layer *l, float *bias);

float get_loss(softmax_layer *l, int *label_in);

#endif
