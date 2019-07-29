#ifndef _LAYER_H_
#define _LAYER_H_

#include <stdbool.h>

#include <cuda.h>
#include <cudnn.h>

#include "execute.h"

//#define USE_CUDNN_FC
#define USE_TENSOR_CORE
//#define USE_TIMER
//#define PRINT_LOSS

#ifdef USE_TIMER
#define START_CNN_TIMER(name) \
static struct timespec st_##name;\
do {\
  LOG(name##_0);\
  synch_device();\
  LOG(name##_1);\
  clock_gettime(CLOCK_MONOTONIC, &st_##name);\
} while (0)

#define STOP_CNN_TIMER(name) \
static struct timespec ed_##name;\
do {\
  LOG(name##_2);\
  synch_device();\
  LOG(name##_3);\
  clock_gettime(CLOCK_MONOTONIC, &ed_##name);\
  l->name += diff_timespec_ms(st_##name, ed_##name);\
} while (0)
#else
#define START_CNN_TIMER(name)
#define STOP_CNN_TIMER(name)
#endif // USE_TIMER

// NETWORK CONNECTION HELPER
#define CONNECT(up, down) \
do {\
  LOG(connect_up_down);\
  assert(bind_buffer2((down).input, (up).output) == 0);\
  assert(bind_buffer3((down).d_input, (up).d_output, (up).d_input, 0) == 0);\
} while (0)

#define CONNECT_FROM_INPUT(in, down) \
do {\
  LOG(connect_in_down);\
  assert(bind_buffer2((down).input, (in).output) == 0);\
  assert(bind_buffer2((down).d_input, (in).d_output) == 0);\
} while (0)

#define CONNECT_WITH_BIAS(up, bias, down) \
do {\
  LOG(connect_up_bias_down);\
  CONNECT(up, down);\
  assert(bind_buffer2((bias).output, (down).input) == 0);\
  assert(bind_buffer2((bias).d_output, (down).d_input) == 0);\
} while (0)

#define CONNECT_FROM_BRANCH(branch, down, j) \
do {\
  LOG(connect_branch_down);\
  assert(bind_buffer2((down).input, (branch).input) == 0);\
  assert(bind_buffer3((down).d_input, (branch).d_output[j], (branch).d_input, j) == 0);\
} while (0)

#define CONNECT_TO_ELT(up, elt, j) \
do {\
  LOG(connect_up_elt);\
  assert(bind_buffer2((elt).input[j], (up).output) == 0);\
  if (j == 0) {\
    assert(bind_buffer3((elt).d_output, (up).d_output, (up).d_input, 0) == 0);\
  }\
  else {\
    assert(bind_buffer2((up).d_output, (elt).d_output) == 0);\
  }\
} while (0)

#define CONNECT_FROM_ELT(elt, down) \
do {\
  LOG(connect_elt_own);\
  assert(bind_buffer2((down).input, (elt).output) == 0);\
  assert(bind_buffer2((down).d_input, (elt).d_output) == 0);\
} while (0)

#define CONNECT_TO_CONCAT(up, concat, j) \
do {\
  LOG(connect_up_concat);\
  assert(bind_buffer2((concat).input[j], (up).output) == 0);\
  if (j == 0) {\
    assert(bind_buffer2((concat).d_output, (up).d_input) == 0);\
  }\
  assert(bind_buffer3((concat).d_input[j], (up).d_output, (concat).d_output, j) == 0);\
} while (0)

#define CONNECT_FROM_CONCAT(concat, down) \
do {\
  LOG(connect_concat_down);\
  assert(bind_buffer2((down).input, (concat).output) == 0);\
  assert(bind_buffer2((down).d_input, (concat).d_output) == 0);\
} while (0)

#define CONNECT_FROM_BRANCH_TO_ELT(branch, elt) \
do {\
  LOG(connect_branch_elt);\
  assert(bind_buffer2((elt).input[0], (branch).input) == 0);\
  assert(bind_buffer3((elt).d_output, (branch).d_output[0], (branch).d_input, 0) == 0);\
} while (0)

#define CONNECT_FROM_BRANCH_TO_CONCAT(branch, concat) \
do {\
  LOG(connect_branch_concat);\
  assert(bind_buffer2((concat).input[0], (branch).input) == 0);\
  assert(bind_buffer2((concat).d_output, (branch).d_input) == 0);\
  assert(bind_buffer3((concat).d_input[0], (branch).d_output[0], (concat).d_output, 0) == 0);\
} while (0)

// PARAM FUNCTIONS
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
  param += set_bn_param(l, param);\
} while (0) 

#define INIT_BIAS(l) \
do {\
  int cnt = (l)->channel;\
  INITIALIZE_CONST(param, cnt, 0.0);\
  param += set_bias(l, param);\
} while (0)

#define SIZE_CONV(l) \
do { sum += param_size_conv(l); } while (0)

#define SIZE_FC(l) \
do { sum += param_size_fc(l); } while (0)

#define SIZE_BN(l) \
do { sum += param_size_bn(l); } while (0)

#define SIZE_BIAS(l) \
do { sum += param_size_bias(l); } while (0)

#define LOAD_CONV(l) \
do { param += set_conv_filter(l, param); } while (0)

#define LOAD_FC(l) \
do { param += set_fc_weight(l, param); } while (0)

#define LOAD_BN(l) \
do { param += set_bn_param(l, param); } while (0) 

#define LOAD_BIAS(l) \
do { param += set_bias(l, param); } while (0)

#define GET_CONV(l) \
do { param += get_conv_filter(l, param); } while (0)

#define GET_FC(l) \
do { param += get_fc_weight(l, param); } while (0)

#define GET_BN(l) \
do { param += get_bn_param(l, param); } while (0)

#define GET_BIAS(l) \
do { param += get_bias(l, param); } while (0)

#define LEN_NAME 256

typedef struct fc_layer_s {
  iterator_t iterator;
  char name[LEN_NAME];

  int batch_size;
  int in, out;

#ifdef USE_CUDNN_FC
  cudnnConvolutionDescriptor_t conv_desc;

  cudnnConvolutionFwdAlgo_t fwd_algo;
  cudnnConvolutionBwdDataAlgo_t bwd_data_algo;
  cudnnConvolutionBwdFilterAlgo_t bwd_filter_algo;
#endif

  gpu_mem input, d_input;
  gpu_mem output, d_output;
  gpu_mem weight, d_weight;
#ifdef USE_CUDNN_FC
  gpu_mem ws_fwd, ws_bwd_data, ws_bwd_filter;
#endif

  float fwd_t, bwd_data_t, bwd_weight_t, bwd_update_t;
} fc_layer;

typedef struct conv_layer_s {
  iterator_t iterator;
  char name[LEN_NAME];

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
  char name[LEN_NAME];

  int batch_size;
  int channel, height, width;

  cudnnBatchNormMode_t mode;

  int nth;
  double eps;

  gpu_mem input, d_input;
  gpu_mem output, d_output;
  gpu_mem scale, d_scale;
  gpu_mem bias, d_bias;
  gpu_mem running_mean, running_var;
  gpu_mem save_mean, save_var;
  
  float fwd_t, bwd_t, bwd_update_t;
} bn_layer;

typedef enum { RELU_T, SIGMOID_T } act_type;

typedef struct act_layer_s {
  iterator_t iterator;
  char name[LEN_NAME];

  int batch_size;
  int channel, height, width;

  act_type type;

  cudnnActivationDescriptor_t act_desc;

  gpu_mem input, d_input;
  gpu_mem output, d_output;
  
  float fwd_t, bwd_t;
} act_layer;

typedef struct concat_layer_s {
  iterator_t iterator;
  char name[LEN_NAME];

  int batch_size;
  int input_channel[FAN_MAX];
  int output_channel, height, width;

  int fan_in;

  gpu_mem input[FAN_MAX], d_input[FAN_MAX];
  gpu_mem output, d_output;
  
  float fwd_t, bwd_t;
} concat_layer;

typedef enum { ADD_T } elt_type;

typedef struct elt_layer_s {
  iterator_t iterator;
  char name[LEN_NAME];

  int batch_size;
  int channel, height, width;

  elt_type type;

  cudnnOpTensorDescriptor_t op_desc;

  gpu_mem input[2];
  gpu_mem output, d_output;
  
  float fwd_t;
} elt_layer;

typedef struct branch_layer_s {
  iterator_t iterator;
  char name[LEN_NAME];

  int batch_size;
  int channel, height, width;

  int fan_out;

  cudnnOpTensorDescriptor_t op_desc;

  gpu_mem input, d_input;
  gpu_mem d_output[FAN_MAX];
  
  float bwd_t;
} branch_layer;

typedef struct bias_layer_s {
  iterator_t iterator;
  char name[LEN_NAME];

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
  char name[LEN_NAME];

  int window_height, window_width;
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
  char name[LEN_NAME];

  int batch_size;
  int channel, height, width;

  gpu_mem output, d_output;
} input_layer;

typedef enum {
  LOG_T, ACCURATE_T
} softmax_type;

typedef struct softmax_layer_s {
  iterator_t iterator;
  char name[LEN_NAME];

  int batch_size;
  int out;

  softmax_type type;

  cudnnOpTensorDescriptor_t op_desc;

  gpu_mem input, d_input;
  gpu_mem output, d_output;
  gpu_mem label;

  float fwd_t, bwd_t;
} softmax_layer;

typedef struct dropout_layer_s {
  iterator_t iterator;
  char name[LEN_NAME];

  int batch_size;
  int out;

  float rate;
  unsigned long long seed;

  cudnnDropoutDescriptor_t dr_desc[MAX_NDEV];

  gpu_mem input, d_input;
  gpu_mem output, d_output;
  gpu_mem rs, st;

  float fwd_t, bwd_t;
} dropout_layer;

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
    int batch_size, int channel, int height, int width);

void init_act_layer(
    act_layer *l, const char *name,
    int batch_size, int channel, int height, int width, act_type type);

void init_pool_layer(
    pool_layer *l, const char *name,
    int batch_size, int filter_height, int filter_width, 
    int pad_height, int pad_width, int stride_height, int stride_width,
    int channel, int input_height, int input_width, pool_type type);

void init_softmax_layer(
    softmax_layer *l, const char *name,
    int batch_size, int out, softmax_type type);

void init_branch_layer(
    branch_layer *l, const char *name,
    int batch_size, int fan_out, int channel, int height, int width);

void init_concat_layer(
    concat_layer *l, const char *name,
    int batch_size, int fan_in, int input_channel[], int height, int width);

void init_dropout_layer(
    dropout_layer *l, const char *name,
    int batch_size, int out, float rate);

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
void train_fwd_dropout_layer(dropout_layer *l);

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
void train_bwd_dropout_layer(dropout_layer *l);

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
void print_time_dropout_layer(dropout_layer *l);

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
void clear_time_dropout_layer(dropout_layer *l);

size_t param_size_conv(conv_layer *l);
size_t param_size_fc(fc_layer *l);
size_t param_size_bn(bn_layer *l);
size_t param_size_bias(bias_layer *l);

int set_conv_filter(conv_layer *l, float *filter);
int set_fc_weight(fc_layer *l, float *weight);
int set_bn_param(bn_layer *l, float *param);
int set_bias(bias_layer *l, float *bias);

int get_conv_filter(conv_layer *l, float *filter);
int get_fc_weight(fc_layer *l, float *weight);
int get_bn_param(bn_layer *l, float *param);
int get_bias(bias_layer *l, float *bias);

float get_loss(softmax_layer *l, int *label_in);

#endif
