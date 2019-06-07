#ifndef _LAYER_H_
#define _LAYER_H_

#include <stdbool.h>

#include <cudnn.h>
#include <cuda.h>

#define DET_CUDNN
//#define TIME_LAYER
//#define PRINT_LOSS
//#define USE_CUBLAS_FC
//#define USE_DROPOUT
//#define CHK_OUTPUT

#define CONNECT(l1, l2) \
do {\
  assert(TENSOR_SIZE((l2).input_desc) == (TENSOR_SIZE((l1).output_desc)));\
  assert(TENSOR_SIZE((l1).d_output_desc) == (TENSOR_SIZE((l2).d_input_desc)));\
  (l2).input = (l1).output;\
  (l1).d_output = (l2).d_input;\
} while (0)

//VGG CONNECTION
#define CONNECT_DIRECT(l1, l2, l3) \
do {\
  assert(TENSOR_SIZE((l3).input_desc) == (TENSOR_SIZE((l1).output_desc)));\
  assert(TENSOR_SIZE((l3).input_desc) == (TENSOR_SIZE((l2).output_desc)));\
  assert(TENSOR_SIZE((l1).d_output_desc) == (TENSOR_SIZE((l3).d_input_desc)));\
  assert(TENSOR_SIZE((l2).d_output_desc) == (TENSOR_SIZE((l3).d_input_desc)));\
  (l3).input = (l2).output = (l1).output;\
  (l1).d_output = (l2).d_output = (l3).d_input;\
} while (0)

//RESNET CONNECTION
//l_elt.d_output must be set first!!
#define CONNECT_DIAMOND_RES(l_branch, l_elt, l_up, l_down) \
do {\
  assert(TENSOR_SIZE((l_elt).input1_desc) == (TENSOR_SIZE((l_branch).input_desc)));\
  assert(TENSOR_SIZE((l_elt).input2_desc) == (TENSOR_SIZE((l_down).output_desc)));\
  assert(TENSOR_SIZE((l_up).input_desc) == (TENSOR_SIZE((l_branch).input_desc)));\
  assert(TENSOR_SIZE((l_branch).d_output_desc) == (TENSOR_SIZE((l_elt).d_output_desc)));\
  assert(TENSOR_SIZE((l_branch).d_output_desc) == (TENSOR_SIZE((l_up).d_input_desc)));\
  assert(TENSOR_SIZE((l_down).d_output_desc) == (TENSOR_SIZE((l_elt).d_output_desc)));\
  (l_elt).input1 = (l_branch).input;\
  (l_elt).input2 = (l_down).output;\
  (l_up).input = (l_branch).input;\
  (l_branch).d_output[0] = (l_elt).d_output;\
  (l_branch).d_output[1] = (l_up).d_input;\
  (l_down).d_output = (l_elt).d_output;\
} while (0)

#define CONNECT_BRANCH_RES(l_branch, l_down1, l_down2) \
do {\
  assert(TENSOR_SIZE((l_down1).input_desc) == (TENSOR_SIZE((l_branch).input_desc)));\
  assert(TENSOR_SIZE((l_down2).input_desc) == (TENSOR_SIZE((l_branch).input_desc)));\
  assert(TENSOR_SIZE((l_branch).d_output_desc) == (TENSOR_SIZE((l_down1).d_input_desc)));\
  assert(TENSOR_SIZE((l_branch).d_output_desc) == (TENSOR_SIZE((l_down2).d_input_desc)));\
  (l_down1).input = (l_branch).input;\
  (l_down2).input = (l_branch).input;\
  (l_branch).d_output[0] = (l_down1).d_input;\
  (l_branch).d_output[1] = (l_down2).d_input;\
} while (0)

//l_elt.d_output must be set first!!
#define CONNECT_ELT(l_up1, l_up2, l_elt) \
do {\
  assert(TENSOR_SIZE((l_elt).input1_desc) == (TENSOR_SIZE((l_up1).output_desc)));\
  assert(TENSOR_SIZE((l_elt).input2_desc) == (TENSOR_SIZE((l_up2).output_desc)));\
  assert(TENSOR_SIZE((l_up1).d_output_desc) == (TENSOR_SIZE((l_elt).d_output_desc)));\
  assert(TENSOR_SIZE((l_up2).d_output_desc) == (TENSOR_SIZE((l_elt).d_output_desc)));\
  (l_elt).input1 = (l_up1).output;\
  (l_elt).input2 = (l_up2).output;\
  (l_up1).d_output = (l_elt).d_output;\
  (l_up2).d_output = (l_elt).d_output;\
} while (0)

//DENSENET CONNECTION
#define CONNECT_DIAMOND_DENSE(l_branch, l_concat, l_up, l_down) \
do {\
  assert(TENSOR_SIZE((l_concat).input_desc[0]) == (TENSOR_SIZE((l_branch).input_desc)));\
  assert(TENSOR_SIZE((l_concat).input_desc[1]) == (TENSOR_SIZE((l_down).output_desc)));\
  assert(TENSOR_SIZE((l_up).input_desc) == (TENSOR_SIZE((l_branch).input_desc)));\
  assert(TENSOR_SIZE((l_branch).d_output_desc) == (TENSOR_SIZE((l_concat).d_input_desc[0])));\
  assert(TENSOR_SIZE((l_branch).d_output_desc) == (TENSOR_SIZE((l_up).d_input_desc)));\
  assert(TENSOR_SIZE((l_down).d_output_desc) == (TENSOR_SIZE((l_concat).d_input_desc[1])));\
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
  assert(TENSOR_SIZE((l_down).input_desc) == (TENSOR_SIZE((l_branch).input_desc)));\
  assert(TENSOR_SIZE((l_branch).d_output_desc) == (TENSOR_SIZE((l_down).d_input_desc)));\
  (l_down).input = (l_branch).input;\
  (l_branch).d_output[i] = (l_down).d_input;\
} while (0)

#define CONNECT_CONCAT_I(l_up, l_concat, i) \
do {\
  assert(TENSOR_SIZE((l_concat).input_desc[i]) == (TENSOR_SIZE((l_up).output_desc)));\
  assert(TENSOR_SIZE((l_up).d_output_desc) == (TENSOR_SIZE((l_concat).d_input_desc[i])));\
  (l_concat).input[i] = (l_up).output;\
  (l_up).d_output = (l_concat).d_input[i];\
} while (0)

#define DUMP_BOTH(l, s, f) \
  DUMP_OUTPUT(l, s, f); DUMP_D_INPUT(l, s, f)

#define DUMP_OUTPUT(l, s, f) \
do {\
  int sz = TENSOR_SIZE((l).output_desc);\
  float *tmp = (float *)malloc(sizeof(float) * sz);\
  chkCUDA(cudaMemcpy(tmp, (l).output, sizeof(float) * sz, cudaMemcpyDeviceToHost));\
  printf("Dump %s output, %d elems\n", s, sz);\
  fwrite(tmp, sizeof(float), sz, f);\
  free(tmp);\
} while (0)

#define DUMP_D_INPUT(l, s, f) \
do {\
  int sz = TENSOR_SIZE((l).d_input_desc);\
  float *tmp = (float *)malloc(sizeof(float) * sz);\
  chkCUDA(cudaMemcpy(tmp, (l).d_input, sizeof(float) * sz, cudaMemcpyDeviceToHost));\
  printf("Dump %s d_input, %d elems\n", s, sz);\
  fwrite(tmp, sizeof(float), sz, f);\
  free(tmp);\
} while (0)

#define LOAD_AND_CHECK_BOTH(l, s, f) \
  LOAD_AND_CHECK_OUTPUT(l, s, f); LOAD_AND_CHECK_D_INPUT(l, s, f)

#define LOAD_AND_CHECK_OUTPUT(l, s, f) \
do {\
  int sz = TENSOR_SIZE((l).output_desc);\
  float *tmp1 = (float *)malloc(sizeof(float) * sz);\
  float *tmp2 = (float *)malloc(sizeof(float) * sz);\
  chkCUDA(cudaMemcpy(tmp1, (l).output, sizeof(float) * sz, cudaMemcpyDeviceToHost));\
  printf("Check %s output, %d elems\n", s, sz);\
  assert(fread(tmp2, sizeof(float), sz, f) == sz);\
  verify(tmp1, tmp2, sz);\
  free(tmp1);\
  free(tmp2);\
} while (0)

#define LOAD_AND_CHECK_D_INPUT(l, s, f) \
do {\
  int sz = TENSOR_SIZE((l).d_input_desc);\
  float *tmp1 = (float *)malloc(sizeof(float) * sz);\
  float *tmp2 = (float *)malloc(sizeof(float) * sz);\
  chkCUDA(cudaMemcpy(tmp1, (l).d_input, sizeof(float) * sz, cudaMemcpyDeviceToHost));\
  printf("Check %s d_input, %d elems\n", s, sz);\
  assert(fread(tmp2, sizeof(float), sz, f) == sz);\
  verify(tmp1, tmp2, sz);\
  free(tmp1);\
  free(tmp2);\
} while (0)

//PARAM INIT
#define INIT_CONV_RES(l) \
do {\
  size_t csz = PSIZE_CONV(l) / sizeof(float);\
  float n_in = (l).filter_height * (l).filter_width * (l).channels_in;\
  float n_out = (l).filter_height * (l).filter_width * (l).channels_out;\
  INITIALIZE_RAND_NORM_SCALE(param, csz, sqrt(2 / (n_out)));\
  param += set_conv_filter(l, param);\
} while (0) 

#define INIT_FC(l) \
do {\
  size_t csz = PSIZE_FC(l) / sizeof(float);\
  float n_in = (l).in;\
  float n_out = (l).out;\
  INITIALIZE_RAND_SCALE(param, csz, sqrt(2 / (n_in + n_out)));\
  param += set_fc_weight(l, param);\
} while (0)

#define INIT_CONV(l) \
do {\
  size_t csz = PSIZE_CONV(l) / sizeof(float);\
  float n_in = (l).filter_height * (l).filter_width * (l).channels_in;\
  float n_out = (l).filter_height * (l).filter_width * (l).channels_out;\
  INITIALIZE_RAND_SCALE(param, csz, sqrt(6 / (n_in + n_out)));\
  param += set_conv_filter(l, param);\
} while (0)

#define INIT_BN(l) \
do {\
  int cnt = (l).channel;\
  INITIALIZE_CONST(param, cnt, 1.0);\
  INITIALIZE_CONST(param + cnt, cnt, 0.0);\
  param += set_bn_vars(l, param);\
} while (0) 

#define INIT_BIAS(l) \
do {\
  int cnt = (l).channel;\
  INITIALIZE_CONST(param, cnt, 0.0);\
  param += set_bias(l, param);\
} while (0)

#define LOAD_CONV_RES(l) \
do {\
  param += set_conv_filter(l, param);\
} while (0) 

#define LOAD_FC(l) \
do {\
  param += set_fc_weight(l, param);\
} while (0)\

#define LOAD_CONV(l) \
do {\
  param += set_conv_filter(l, param);\
} while (0)\

#define LOAD_BN(l) \
do {\
  param += set_bn_vars(l, param);\
} while (0) 

#define LOAD_BIAS(l) \
do {\
  param += set_bias(l, param);\
} while (0)

#define GET_FC(l) \
do {\
  param += get_fc_weight(l, param);\
} while (0)

#define GET_CONV(l) \
do {\
  param += get_conv_filter(l, param);\
} while (0)

#define GET_BN(l) \
do {\
  param += get_bn_vars(l, param);\
} while (0)

#define GET_BIAS(l) \
do {\
  param += get_bias(l, param);\
} while (0)

typedef struct fc_layer_s {
  cudnnHandle_t cudnn;

  int batch_size, in, out;

  cudnnTensorDescriptor_t input_desc;
  cudnnTensorDescriptor_t d_input_desc;

  cudnnTensorDescriptor_t output_desc;
  cudnnTensorDescriptor_t d_output_desc;

  cudnnConvolutionDescriptor_t conv_desc;
  cudnnFilterDescriptor_t filter_desc;
  cudnnConvolutionFwdAlgo_t fwd_algo;
  cudnnFilterDescriptor_t d_filter_desc;
  cudnnConvolutionBwdDataAlgo_t bwd_data_algo;
  cudnnConvolutionBwdFilterAlgo_t bwd_filter_algo;

  size_t conv_workspace_bytes;
  size_t conv_workspace_bwd_data_bytes;
  size_t conv_workspace_bwd_filter_bytes;

  float *filter;
  float *d_filter;

  float *input;
  float *output;

  float *d_input;
  float *d_output;

  float fwd_t, bwd_data_t, bwd_weight_t, bwd_update_t;
} fc_layer;

typedef struct conv_layer_s {
  cudnnHandle_t cudnn;

  bool is_training;

  int filter_height, filter_width;
  int pad_height, pad_width;
  int stride_x, stride_y;
  int channels_out, channels_in;
  int output_height, output_width;

  cudnnTensorDescriptor_t input_desc;
  cudnnTensorDescriptor_t d_input_desc;

  cudnnTensorDescriptor_t output_desc;
  cudnnTensorDescriptor_t d_output_desc;

  cudnnConvolutionDescriptor_t conv_desc;
  cudnnFilterDescriptor_t filter_desc;
  cudnnConvolutionFwdAlgo_t fwd_algo;

  cudnnFilterDescriptor_t d_filter_desc;
  cudnnConvolutionBwdDataAlgo_t bwd_data_algo;
  cudnnConvolutionBwdFilterAlgo_t bwd_filter_algo;

  float *input;
  float *output;

  float *d_input;
  float *d_output;

  float *filter;
  float *d_filter;

  size_t conv_workspace_bytes;
  size_t conv_workspace_bwd_data_bytes;
  size_t conv_workspace_bwd_filter_bytes;

  float fwd_t, bwd_data_t, bwd_filter_t, bwd_update_t;
} conv_layer;

typedef struct bn_layer_s { 
  cudnnHandle_t cudnn;

  bool is_training;

  int channel;
  int height, width;
  double bn_eafactor;
  double bn_epsilon;

  cudnnTensorDescriptor_t input_desc;
  cudnnTensorDescriptor_t d_input_desc;

  cudnnTensorDescriptor_t output_desc;
  cudnnTensorDescriptor_t d_output_desc;

  cudnnTensorDescriptor_t bn_desc;
  cudnnTensorDescriptor_t d_bn_desc;

  float *input;
  float *d_input;

  float *output;
  float *d_output;

  float *bn_scale;
  float *bn_bias;
  float *bn_result_running_mean;
  float *bn_result_running_var;
  float *bn_result_save_mean;
  float *bn_result_save_var;

  float *d_bn_scale;
  float *d_bn_bias;
  
  float fwd_t, bwd_t, bwd_update_t;
} bn_layer;

typedef enum {
  relu
} ACT_TYPE;

typedef struct act_layer_s {
  cudnnHandle_t cudnn;

  bool is_training;

  ACT_TYPE type;

  int channel;
  int height, width;

  cudnnTensorDescriptor_t input_desc;
  cudnnTensorDescriptor_t d_input_desc;

  cudnnTensorDescriptor_t output_desc; 
  cudnnTensorDescriptor_t d_output_desc; 

  cudnnActivationDescriptor_t act_desc;

  float *input;
  float *d_input;

  float *output;
  float *d_output;
  
  float fwd_t, bwd_t;
} act_layer;

typedef enum {
  addition
} ELT_TYPE;

#define MAX_IN 16
typedef struct concat_layer_s {
  cudnnHandle_t cudnn;

  int in_cnt;

  int batch;
  int channel_in[MAX_IN];
  int channel;
  int height, width;

  cudnnTensorDescriptor_t input_desc[MAX_IN];
  cudnnTensorDescriptor_t d_input_desc[MAX_IN];

  cudnnTensorDescriptor_t output_desc; 
  cudnnTensorDescriptor_t d_output_desc; 

  float *input[MAX_IN];
  float *d_input[MAX_IN];
  float *output;
  float *d_output;
  
  float fwd_t, bwd_t;
} concat_layer;

typedef struct elt_layer_s {
  cudnnHandle_t cudnn;

  bool is_training;

  ELT_TYPE type;

  int batch;
  int channel;
  int height, width;

  cudnnTensorDescriptor_t input1_desc;
  cudnnTensorDescriptor_t input2_desc;

  cudnnTensorDescriptor_t output_desc; 
  cudnnTensorDescriptor_t d_output_desc; 

  cudnnOpTensorDescriptor_t op_desc;

  float *input1;
  float *input2;
  float *output;
  float *d_output;
  
  float fwd_t, bwd_t;
} elt_layer;

#define MAX_OUT 16
typedef struct branch_layer_s {
  cudnnHandle_t cudnn;

  bool is_training;

  int out_cnt;

  int batch;
  int channel;
  int height, width;

  cudnnTensorDescriptor_t input_desc;
  cudnnTensorDescriptor_t d_input_desc;
  cudnnTensorDescriptor_t d_output_desc; 

  cudnnOpTensorDescriptor_t op_desc;

  float *input;
  float *d_input;
  float *d_output[MAX_OUT];
  
  float fwd_t, bwd_t;
} branch_layer;

typedef struct bias_layer_s {
  cudnnHandle_t cudnn;

  bool is_training;

  int channel;
  int height, width;

  cudnnTensorDescriptor_t output_desc; 
  cudnnTensorDescriptor_t d_output_desc; 

  cudnnTensorDescriptor_t bias_desc; 
  cudnnTensorDescriptor_t d_bias_desc; 

  float *output;
  float *d_output;
  
  float *bias;
  float *d_bias;
  
  float fwd_t, bwd_t, bwd_update_t;
} bias_layer;

typedef enum {
  max, average
} POOL_TYPE;

typedef struct pool_layer_s {
  cudnnHandle_t cudnn;

  POOL_TYPE type;

  int channel;
  int height, width;
  int filter_height, filter_width;
  int pad_height, pad_width;
  int stride_x, stride_y;
  int height_output, width_output;

  cudnnTensorDescriptor_t input_desc;
  cudnnTensorDescriptor_t d_input_desc;

  cudnnTensorDescriptor_t output_desc;
  cudnnTensorDescriptor_t d_output_desc;

  cudnnPoolingDescriptor_t pooling_desc;

  float *input;
  float *output;

  float *d_input;
  float *d_output;

  float fwd_t, bwd_t;
} pool_layer;

typedef struct input_layer_s {
  cudnnHandle_t cudnn;
  int channel;
  int height, width;

  cudnnTensorDescriptor_t output_desc;
  cudnnTensorDescriptor_t d_output_desc;

  float *output;
  float *d_output;
} input_layer;

typedef struct softmax_layer_s {
  cudnnHandle_t cudnn;
 
  bool is_training;

  int batch_size;
  int out;

  cudnnTensorDescriptor_t input_desc;
  cudnnTensorDescriptor_t output_desc;
  cudnnTensorDescriptor_t d_input_desc;

  cudnnOpTensorDescriptor_t op_desc;

  float *input;
  float *d_input;
  float *output;
  float *label;
  int *label_in;
  
  float fwd_t, bwd_t;
} softmax_layer;

void init_input_layer(
    input_layer *l, cudnnHandle_t cudnn,
    int batch, int channel, int height, int width);

void init_elt_layer(
    elt_layer *l, cudnnHandle_t cudnn,
    int batch, int channel, int height, int width, ELT_TYPE type);

void init_bias_layer(
    bias_layer *l, cudnnHandle_t cudnn,
    int batch, int channel, int height, int width);

void init_conv_layer(
    conv_layer *l, cudnnHandle_t cudnn,
    int batch, int filter_height, int filter_width, int pad_height, int pad_width,
    int stride_x, int stride_y, int in, int out, int height, int width);

void init_conv_workspace();

void init_fc_layer(
    fc_layer *l, cudnnHandle_t cudnn,
    int batch_size, int in, int out);

void init_bn_layer(
    bn_layer *l, cudnnHandle_t cudnn,
    int batch, int channel, int height, int width);

void init_act_layer(
    act_layer *l, cudnnHandle_t cudnn,
    int batch, int channel, int height, int width);

void init_pool_layer(
    pool_layer *l, cudnnHandle_t cudnn,
    int batch, int filter_height, int filter_width, 
    int pad_height, int pad_width,
    int stride_x, int stride_y, int channel, int height, int width, POOL_TYPE type);

void init_softmax_layer(
    softmax_layer *fcl, cudnnHandle_t cudnn,
    int batch, int out);

void init_branch_layer(
    branch_layer *l, cudnnHandle_t cudnn,
    int batch, int out_cnt, int channel, int height, int width);

void init_concat_layer(
    concat_layer *l, cudnnHandle_t cudnn,
    int batch, int in_cnt, int *channels_in, int height, int width);

void train_fwd_conv_layer(conv_layer *l);
void train_fwd_fc_layer(fc_layer *l);
void train_fwd_bn_layer(bn_layer *l);
void train_fwd_act_layer(act_layer * l);
void train_fwd_pool_layer(pool_layer * l);
void train_fwd_elt_layer(elt_layer *l);
void train_fwd_softmax_layer(softmax_layer *l);
void train_fwd_branch_layer(branch_layer *l);
void train_fwd_bias_layer(bias_layer *l);
void train_fwd_concat_layer(concat_layer *l);

void train_bwd_conv_layer(conv_layer *l);
void train_bwd_fc_layer(fc_layer *l);
void train_bwd_bn_layer(bn_layer *l);
void train_bwd_bn_res_layer(bn_layer *l);
void train_bwd_act_layer(act_layer * l);
void train_bwd_pool_layer(pool_layer * l);
void train_bwd_elt_layer(elt_layer *l);
void train_bwd_softmax_layer(softmax_layer *l);
void train_bwd_branch_layer(branch_layer *l);
void train_bwd_bias_layer(bias_layer *l);
void train_bwd_concat_layer(concat_layer *l);

void print_time_conv_layer(conv_layer *l, char *name);
void print_time_fc_layer(fc_layer *l, char *name);
void print_time_bn_layer(bn_layer *l, char *name);
void print_time_act_layer(act_layer *l, char *name);
void print_time_pool_layer(pool_layer * l, char *name);
void print_time_elt_layer(elt_layer *l, char *name);
void print_time_softmax_layer(softmax_layer *l, char *name);
void print_time_branch_layer(branch_layer *l, char *name);
void print_time_bias_layer(bias_layer *l, char *name);
void print_time_concat_layer(concat_layer *l, char *name);

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

int set_conv_filter(conv_layer l, float *filter);
int get_conv_filter(conv_layer l, float *filter);
int set_fc_weight(fc_layer l, float *weight);
int get_fc_weight(fc_layer l, float *weight);
int get_bn_vars(bn_layer l, float *bn);
int set_bn_vars(bn_layer l, float *bn);
int get_bias(bias_layer l, float *bias);
int set_bias(bias_layer l, float *bias);

float get_loss(softmax_layer *l, int *label_in);

static inline size_t PSIZE_CONV(conv_layer l)
{
  return sizeof(float) * l.filter_height * l.filter_width * l.channels_in * l.channels_out;
}

static inline size_t PSIZE_FC(fc_layer l)
{
  return sizeof(float) * l.in * l.out;
}

static inline size_t PSIZE_BN(bn_layer l)
{
  return sizeof(float) * l.channel * 2;
}

static inline size_t PSIZE_BIAS(bias_layer l)
{
  return sizeof(float) * l.channel; 
}

#endif
