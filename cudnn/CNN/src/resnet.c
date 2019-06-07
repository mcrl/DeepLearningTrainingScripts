#include <stdio.h>
#include <string.h>
#include <stdbool.h>
#include <assert.h>
#include <math.h>
#include <time.h>

#include <cuda.h>
#include <cuda_profiler_api.h>
#include <cudnn.h>
#include <cublas_v2.h>

#include "cnn.h"
#include "cnn_cuda.h"

#include "params.h"
#include "utils.h"

#include "layer.h"

static float one = 1.0f;
static float zero = 0.0f;

cudnnHandle_t cudnn;
cublasHandle_t cublas;

cudnnTensorDescriptor_t grads_desc;
cudnnTensorDescriptor_t probs_desc;

float *p;
float *d;
float *buf_d = NULL;

int off_d;

int *labels;

#define RESNET50
//#define RESNET101
//#define RESNET152

#ifdef RESNET50
#define B0 3
#define B1 4
#define B2 6
#define B3 4
#endif
#ifdef RESNET101
#define B0 3
#define B1 4
#define B2 23
#define B3 4
#endif
#ifdef RESNET152
#define B0 3
#define B1 8
#define B2 36
#define B3 4
#endif

typedef struct resnet_s {
  input_layer input;
  conv_layer conv1;
  bn_layer conv1_bn;
  act_layer conv1_relu;
  pool_layer pool1;

  conv_layer conv2_branch;
  bn_layer conv2_branch_bn;
  conv_layer conv3_branch;
  bn_layer conv3_branch_bn;
  conv_layer conv4_branch;
  bn_layer conv4_branch_bn;
  conv_layer conv5_branch;
  bn_layer conv5_branch_bn;

  conv_layer conv2[B0][3];
  bn_layer conv2_bn[B0][3];
  act_layer conv2_relu[B0][3]; // Third ReLU is after Eltwise addition
  elt_layer conv2_add[B0];
  branch_layer branch2[B0];

  conv_layer conv3[B1][3];
  bn_layer conv3_bn[B1][3];
  act_layer conv3_relu[B1][3]; // Third ReLU is after Eltwise addition
  elt_layer conv3_add[B1];
  branch_layer branch3[B1];

  conv_layer conv4[B2][3];
  bn_layer conv4_bn[B2][3];
  act_layer conv4_relu[B2][3]; // Third ReLU is after Eltwise addition
  elt_layer conv4_add[B2];
  branch_layer branch4[B2];

  conv_layer conv5[B3][3];
  bn_layer conv5_bn[B3][3];
  act_layer conv5_relu[B3][3]; // Third ReLU is after Eltwise addition
  elt_layer conv5_add[B3];
  branch_layer branch5[B3];

  pool_layer pool2;
  
  conv_layer fc;
  bias_layer bias;
  softmax_layer softmax;
} resnet;

resnet net;

bool is_initiated = false;

void params_modify()
{
}

void resnet_init(int batch_size)
{
  chkCUDNN(cudnnCreate(&cudnn));
  chkCUBLAS(cublasCreate(&cublas));
  srand(params.seed);

  init_input_layer(&net.input, cudnn, batch_size, 3, 224, 224);
  init_conv_layer(&net.conv1, cudnn, batch_size, 7, 7, 3, 3, 2, 2, 3, 64, 224, 224);
  init_bn_layer(&net.conv1_bn, cudnn, batch_size, 64, 112, 112); 
  init_act_layer(&net.conv1_relu, cudnn, batch_size, 64, 112, 112);
  init_pool_layer(&net.pool1, cudnn, batch_size, 3, 3, 1, 1, 2, 2, 64, 112, 112, max);

  init_conv_layer(&net.conv2_branch, cudnn, batch_size, 1, 1, 0, 0, 1, 1, 64, 256, 56, 56);
  init_bn_layer(&net.conv2_branch_bn, cudnn, batch_size, 256, 56, 56);
  int prev_channel = 64;

  for (int i = 0; i < B0; i++) {
    init_branch_layer(&net.branch2[i], cudnn, batch_size, 2, prev_channel, 56, 56);
    init_conv_layer(&net.conv2[i][0], cudnn, batch_size, 1, 1, 0, 0, 1, 1, prev_channel, 64, 56, 56);
    init_bn_layer(&net.conv2_bn[i][0], cudnn, batch_size, 64, 56, 56);
    init_act_layer(&net.conv2_relu[i][0], cudnn, batch_size, 64, 56, 56);
    init_conv_layer(&net.conv2[i][1], cudnn, batch_size, 3, 3, 1, 1, 1, 1, 64, 64, 56, 56);
    init_bn_layer(&net.conv2_bn[i][1], cudnn, batch_size, 64, 56, 56);
    init_act_layer(&net.conv2_relu[i][1], cudnn, batch_size, 64, 56, 56);
    init_conv_layer(&net.conv2[i][2], cudnn, batch_size, 1, 1, 0, 0, 1, 1, 64, 256, 56, 56);
    init_bn_layer(&net.conv2_bn[i][2], cudnn, batch_size, 256, 56, 56);
    init_act_layer(&net.conv2_relu[i][2], cudnn, batch_size, 256, 56, 56);
    init_elt_layer(&net.conv2_add[i], cudnn, batch_size, 256, 56, 56, addition);
    prev_channel = 256;
  }

  init_conv_layer(&net.conv3_branch, cudnn, batch_size, 1, 1, 0, 0, 2, 2, 256, 512, 56, 56);
  init_bn_layer(&net.conv3_branch_bn, cudnn, batch_size, 512, 28, 28);

  for (int i = 0; i < B1; i++) {
    if (i == 0) {
      init_branch_layer(&net.branch3[i], cudnn, batch_size, 2, prev_channel, 56, 56);
      init_conv_layer(&net.conv3[i][0], cudnn, batch_size, 1, 1, 0, 0, 2, 2, prev_channel, 128, 56, 56);
    }
    else {
      init_branch_layer(&net.branch3[i], cudnn, batch_size, 2, prev_channel, 28, 28);
      init_conv_layer(&net.conv3[i][0], cudnn, batch_size, 1, 1, 0, 0, 1, 1, prev_channel, 128, 28, 28);
    }
    init_bn_layer(&net.conv3_bn[i][0], cudnn, batch_size, 128, 28, 28);
    init_act_layer(&net.conv3_relu[i][0], cudnn, batch_size, 128, 28, 28);
    init_conv_layer(&net.conv3[i][1], cudnn, batch_size, 3, 3, 1, 1, 1, 1, 128, 128, 28, 28);
    init_bn_layer(&net.conv3_bn[i][1], cudnn, batch_size, 128, 28, 28);
    init_act_layer(&net.conv3_relu[i][1], cudnn, batch_size, 128, 28, 28);
    init_conv_layer(&net.conv3[i][2], cudnn, batch_size, 1, 1, 0, 0, 1, 1, 128, 512, 28, 28);
    init_bn_layer(&net.conv3_bn[i][2], cudnn, batch_size, 512, 28, 28);
    init_act_layer(&net.conv3_relu[i][2], cudnn, batch_size, 512, 28, 28);
    init_elt_layer(&net.conv3_add[i], cudnn, batch_size, 512, 28, 28, addition);
    prev_channel = 512;
  }

  init_conv_layer(&net.conv4_branch, cudnn, batch_size, 1, 1, 0, 0, 2, 2, 512, 1024, 28, 28);
  init_bn_layer(&net.conv4_branch_bn, cudnn, batch_size, 1024, 14, 14);

  for (int i = 0; i < B2; i++) {
    if (i == 0) {
      init_branch_layer(&net.branch4[i], cudnn, batch_size, 2, prev_channel, 28, 28);
      init_conv_layer(&net.conv4[i][0], cudnn, batch_size, 1, 1, 0, 0, 2, 2, prev_channel, 256, 28, 28);
    }
    else {
      init_branch_layer(&net.branch4[i], cudnn, batch_size, 2, prev_channel, 14, 14);
      init_conv_layer(&net.conv4[i][0], cudnn, batch_size, 1, 1, 0, 0, 1, 1, prev_channel, 256, 14, 14);
    }
    init_bn_layer(&net.conv4_bn[i][0], cudnn, batch_size, 256, 14, 14);
    init_act_layer(&net.conv4_relu[i][0], cudnn, batch_size, 256, 14, 14);
    init_conv_layer(&net.conv4[i][1], cudnn, batch_size, 3, 3, 1, 1, 1, 1, 256, 256, 14, 14);
    init_bn_layer(&net.conv4_bn[i][1], cudnn, batch_size, 256, 14, 14);
    init_act_layer(&net.conv4_relu[i][1], cudnn, batch_size, 256, 14, 14);
    init_conv_layer(&net.conv4[i][2], cudnn, batch_size, 1, 1, 0, 0, 1, 1, 256, 1024, 14, 14);
    init_bn_layer(&net.conv4_bn[i][2], cudnn, batch_size, 1024, 14, 14);
    init_act_layer(&net.conv4_relu[i][2], cudnn, batch_size, 1024, 14, 14);
    init_elt_layer(&net.conv4_add[i], cudnn, batch_size, 1024, 14, 14, addition);
    prev_channel = 1024;
  }

  init_conv_layer(&net.conv5_branch, cudnn, batch_size, 1, 1, 0, 0, 2, 2, 1024, 2048, 14, 14);
  init_bn_layer(&net.conv5_branch_bn, cudnn, batch_size, 2048, 7, 7);

  for (int i = 0; i < B3; i++) {
    if (i == 0) {
      init_branch_layer(&net.branch5[i], cudnn, batch_size, 2, prev_channel, 14, 14);
      init_conv_layer(&net.conv5[i][0], cudnn, batch_size, 1, 1, 0, 0, 2, 2, prev_channel, 512, 14, 14);
    }
    else {
      init_branch_layer(&net.branch5[i], cudnn, batch_size, 2, prev_channel, 7, 7);
      init_conv_layer(&net.conv5[i][0], cudnn, batch_size, 1, 1, 0, 0, 1, 1, prev_channel, 512, 7, 7);
    }
    init_bn_layer(&net.conv5_bn[i][0], cudnn, batch_size, 512, 7, 7);
    init_act_layer(&net.conv5_relu[i][0], cudnn, batch_size, 512, 7, 7);
    init_conv_layer(&net.conv5[i][1], cudnn, batch_size, 3, 3, 1, 1, 1, 1, 512, 512, 7, 7);
    init_bn_layer(&net.conv5_bn[i][1], cudnn, batch_size, 512, 7, 7);
    init_act_layer(&net.conv5_relu[i][1], cudnn, batch_size, 512, 7, 7);
    init_conv_layer(&net.conv5[i][2], cudnn, batch_size, 1, 1, 0, 0, 1, 1, 512, 2048, 7, 7);
    init_bn_layer(&net.conv5_bn[i][2], cudnn, batch_size, 2048, 7, 7);
    init_act_layer(&net.conv5_relu[i][2], cudnn, batch_size, 2048, 7, 7);
    init_elt_layer(&net.conv5_add[i], cudnn, batch_size, 2048, 7, 7, addition);
    prev_channel = 2048;
  }

  init_pool_layer(&net.pool2, cudnn, batch_size, 7, 7, 0, 0, 1, 1, 2048, 7, 7, average);
  init_conv_layer(&net.fc, cudnn, batch_size, 1, 1, 0, 0, 1, 1, 2048, 1000, 1, 1);
  init_bias_layer(&net.bias, cudnn, batch_size, 1000, 1, 1);
  init_softmax_layer(&net.softmax, cudnn, batch_size, 1000);

  init_conv_workspace();

  is_initiated = true;
}

size_t resnet_get_param_size()
{
  size_t sum = 0;

  sum += PSIZE_CONV(net.conv1);
  sum += PSIZE_BN(net.conv1_bn);
  sum += PSIZE_CONV(net.conv2_branch);
  sum += PSIZE_BN(net.conv2_branch_bn);
  sum += PSIZE_CONV(net.conv3_branch);
  sum += PSIZE_BN(net.conv3_branch_bn);
  sum += PSIZE_CONV(net.conv4_branch);
  sum += PSIZE_BN(net.conv4_branch_bn);
  sum += PSIZE_CONV(net.conv5_branch);
  sum += PSIZE_BN(net.conv5_branch_bn);

  for (int i = 0; i < B0; i++)
    for (int j = 0; j < 3; j++) {
      sum += PSIZE_CONV(net.conv2[i][j]);
      sum += PSIZE_BN(net.conv2_bn[i][j]);
    }

  for (int i = 0; i < B1; i++)
    for (int j = 0; j < 3; j++) {
      sum += PSIZE_CONV(net.conv3[i][j]);
      sum += PSIZE_BN(net.conv3_bn[i][j]);
    }

  for (int i = 0; i < B2; i++)
    for (int j = 0; j < 3; j++) {
      sum += PSIZE_CONV(net.conv4[i][j]);
      sum += PSIZE_BN(net.conv4_bn[i][j]);
    }

  for (int i = 0; i < B3; i++)
    for (int j = 0; j < 3; j++) {
      sum += PSIZE_CONV(net.conv5[i][j]);
      sum += PSIZE_BN(net.conv5_bn[i][j]);
    }

  sum += PSIZE_CONV(net.fc);
  sum += PSIZE_BIAS(net.bias);

  return sum;
}

void resnet_load_param(float *param)
{
  LOAD_CONV_RES(net.conv1);
  LOAD_BN(net.conv1_bn);
  LOAD_CONV_RES(net.conv2_branch);
  LOAD_BN(net.conv2_branch_bn);
  LOAD_CONV_RES(net.conv3_branch);
  LOAD_BN(net.conv3_branch_bn);
  LOAD_CONV_RES(net.conv4_branch);
  LOAD_BN(net.conv4_branch_bn);
  LOAD_CONV_RES(net.conv5_branch);
  LOAD_BN(net.conv5_branch_bn);

  for (int i = 0; i < B0; i++)
    for (int j = 0; j < 3; j++) {
      LOAD_CONV_RES(net.conv2[i][j]);
      LOAD_BN(net.conv2_bn[i][j]);
    }

  for (int i = 0; i < B1; i++)
    for (int j = 0; j < 3; j++) {
      LOAD_CONV_RES(net.conv3[i][j]);
      LOAD_BN(net.conv3_bn[i][j]);
    }

  for (int i = 0; i < B2; i++)
    for (int j = 0; j < 3; j++) {
      LOAD_CONV_RES(net.conv4[i][j]);
      LOAD_BN(net.conv4_bn[i][j]);
    }

  for (int i = 0; i < B3; i++)
    for (int j = 0; j < 3; j++) {
      LOAD_CONV_RES(net.conv5[i][j]);
      LOAD_BN(net.conv5_bn[i][j]);
    }

  LOAD_CONV(net.fc);
  LOAD_BIAS(net.bias);
}

void resnet_set_param(float *param)
{
  INIT_CONV_RES(net.conv1);
  INIT_BN(net.conv1_bn);
  INIT_CONV_RES(net.conv2_branch);
  INIT_BN(net.conv2_branch_bn);
  INIT_CONV_RES(net.conv3_branch);
  INIT_BN(net.conv3_branch_bn);
  INIT_CONV_RES(net.conv4_branch);
  INIT_BN(net.conv4_branch_bn);
  INIT_CONV_RES(net.conv5_branch);
  INIT_BN(net.conv5_branch_bn);

  for (int i = 0; i < B0; i++)
    for (int j = 0; j < 3; j++) {
      INIT_CONV_RES(net.conv2[i][j]);
      INIT_BN(net.conv2_bn[i][j]);
    }

  for (int i = 0; i < B1; i++)
    for (int j = 0; j < 3; j++) {
      INIT_CONV_RES(net.conv3[i][j]);
      INIT_BN(net.conv3_bn[i][j]);
    }

  for (int i = 0; i < B2; i++)
    for (int j = 0; j < 3; j++) {
      INIT_CONV_RES(net.conv4[i][j]);
      INIT_BN(net.conv4_bn[i][j]);
    }

  for (int i = 0; i < B3; i++)
    for (int j = 0; j < 3; j++) {
      INIT_CONV_RES(net.conv5[i][j]);
      INIT_BN(net.conv5_bn[i][j]);
    }

  INIT_CONV(net.fc);
  INIT_BIAS(net.bias);
}

void resnet_get_param(float *param)
{
  param += get_conv_filter(net.conv1, param);
  param += get_bn_vars(net.conv1_bn, param);
  param += get_conv_filter(net.conv2_branch, param);
  param += get_bn_vars(net.conv2_branch_bn, param);
  param += get_conv_filter(net.conv3_branch, param);
  param += get_bn_vars(net.conv3_branch_bn, param);
  param += get_conv_filter(net.conv4_branch, param);
  param += get_bn_vars(net.conv4_branch_bn, param);
  param += get_conv_filter(net.conv5_branch, param);
  param += get_bn_vars(net.conv5_branch_bn, param);

  for (int i = 0; i < B0; i++)
    for (int j = 0; j < 3; j++) {
      param += get_conv_filter(net.conv2[i][j], param);
      param += get_bn_vars(net.conv2_bn[i][j], param);
    }

  for (int i = 0; i < B1; i++)
    for (int j = 0; j < 3; j++) {
      param += get_conv_filter(net.conv3[i][j], param);
      param += get_bn_vars(net.conv3_bn[i][j], param);
    }

  for (int i = 0; i < B2; i++)
    for (int j = 0; j < 3; j++) {
      param += get_conv_filter(net.conv4[i][j], param);
      param += get_bn_vars(net.conv4_bn[i][j], param);
    }

  for (int i = 0; i < B3; i++)
    for (int j = 0; j < 3; j++) {
      param += get_conv_filter(net.conv5[i][j], param);
      param += get_bn_vars(net.conv5_bn[i][j], param);
    }

  param += get_conv_filter(net.fc, param);
  param += get_bias(net.bias, param);
}

void resnet_copy_input(int batch_size, float * data_in, int * label_in)
{
  size_t input_size = sizeof(float) * batch_size * params.width * params.height * params.channel;

  chkCUDA(cudaMemcpy(net.input.output, data_in, input_size, cudaMemcpyHostToDevice)); 
  chkCUDA(cudaMemcpy(net.softmax.label_in, label_in, batch_size * sizeof(int), cudaMemcpyHostToDevice)); 
  cuda_set_label(batch_size, 1000, net.softmax.label_in, net.softmax.label);
}

void resnet_forward()
{
  train_fwd_conv_layer(&net.conv1);
  train_fwd_bn_layer(&net.conv1_bn);
  train_fwd_act_layer(&net.conv1_relu);
  train_fwd_pool_layer(&net.pool1);

  for (int i = 0; i < B0; i++) {
    for (int j = 0; j < 3; j++) {
      if (j == 0) {
        if (i == 0) {
          train_fwd_branch_layer(&net.branch2[i]);
          train_fwd_conv_layer(&net.conv2_branch);
          train_fwd_bn_layer(&net.conv2_branch_bn);
        }
        else {
          train_fwd_branch_layer(&net.branch2[i]);
        }
      }

      train_fwd_conv_layer(&net.conv2[i][j]);
      train_fwd_bn_layer(&net.conv2_bn[i][j]);

      if (j == 2) {
        train_fwd_elt_layer(&net.conv2_add[i]);
      }

      train_fwd_act_layer(&net.conv2_relu[i][j]);
    }
  }

  for (int i = 0; i < B1; i++) {
    for (int j = 0; j < 3; j++) {
      if (j == 0) {
        if (i == 0) {
          train_fwd_branch_layer(&net.branch3[i]);
          train_fwd_conv_layer(&net.conv3_branch);
          train_fwd_bn_layer(&net.conv3_branch_bn);
        }
        else {
          train_fwd_branch_layer(&net.branch3[i]);
        }
      }

      train_fwd_conv_layer(&net.conv3[i][j]);
      train_fwd_bn_layer(&net.conv3_bn[i][j]);

      if (j == 2) {
        train_fwd_elt_layer(&net.conv3_add[i]);
      }

      train_fwd_act_layer(&net.conv3_relu[i][j]);
    }
  }

  for (int i = 0; i < B2; i++) {
    for (int j = 0; j < 3; j++) {
      if (j == 0) {
        if (i == 0) {
          train_fwd_branch_layer(&net.branch4[i]);
          train_fwd_conv_layer(&net.conv4_branch);
          train_fwd_bn_layer(&net.conv4_branch_bn);
        }
        else {
          train_fwd_branch_layer(&net.branch4[i]);
        }
      }

      train_fwd_conv_layer(&net.conv4[i][j]);
      train_fwd_bn_layer(&net.conv4_bn[i][j]);

      if (j == 2) {
        train_fwd_elt_layer(&net.conv4_add[i]);
      }

      train_fwd_act_layer(&net.conv4_relu[i][j]);
    }
  }
 
  for (int i = 0; i < B3; i++) {
    for (int j = 0; j < 3; j++) {
      if (j == 0) {
        if (i == 0) {
          train_fwd_branch_layer(&net.branch5[i]);
          train_fwd_conv_layer(&net.conv5_branch);
          train_fwd_bn_layer(&net.conv5_branch_bn);
        }
        else {
          train_fwd_branch_layer(&net.branch5[i]);
        }
      }

      train_fwd_conv_layer(&net.conv5[i][j]);
      train_fwd_bn_layer(&net.conv5_bn[i][j]);

      if (j == 2) {
        train_fwd_elt_layer(&net.conv5_add[i]);
      }

      train_fwd_act_layer(&net.conv5_relu[i][j]);
    }
  }

  train_fwd_pool_layer(&net.pool2);
  train_fwd_conv_layer(&net.fc);
  train_fwd_bias_layer(&net.bias);
  train_fwd_softmax_layer(&net.softmax);
}

void resnet_backward()
{
  train_bwd_softmax_layer(&net.softmax);
  train_bwd_bias_layer(&net.bias);
  train_bwd_conv_layer(&net.fc);
  train_bwd_pool_layer(&net.pool2);

  for (int i = B3-1; i >= 0; i--) {
    for (int j = 2; j >= 0; j--) {
      train_bwd_act_layer(&net.conv5_relu[i][j]);

      if (j == 2) {
        train_bwd_elt_layer(&net.conv5_add[i]);
      }

      train_bwd_bn_layer(&net.conv5_bn[i][j]);
      train_bwd_conv_layer(&net.conv5[i][j]);

      if (j == 0) {
        if (i == 0) {
          train_bwd_bn_layer(&net.conv5_branch_bn);
          train_bwd_conv_layer(&net.conv5_branch);
          train_bwd_branch_layer(&net.branch5[i]);
        }
        else {
          train_bwd_branch_layer(&net.branch5[i]);
        }
      }
    }
  }

  for (int i = B2-1; i >= 0; i--) {
    for (int j = 2; j >= 0; j--) {
      train_bwd_act_layer(&net.conv4_relu[i][j]);

      if (j == 2) {
        train_bwd_elt_layer(&net.conv4_add[i]);
      }

      train_bwd_bn_layer(&net.conv4_bn[i][j]);
      train_bwd_conv_layer(&net.conv4[i][j]);

      if (j == 0) {
        if (i == 0) {
          train_bwd_bn_layer(&net.conv4_branch_bn);
          train_bwd_conv_layer(&net.conv4_branch);
          train_bwd_branch_layer(&net.branch4[i]);
        }
        else {
          train_bwd_branch_layer(&net.branch4[i]);
        }
      }
    }
  }

  for (int i = B1-1; i >= 0; i--) {
    for (int j = 2; j >= 0; j--) {
      train_bwd_act_layer(&net.conv3_relu[i][j]);

      if (j == 2) {
        train_bwd_elt_layer(&net.conv3_add[i]);
      }

      train_bwd_bn_layer(&net.conv3_bn[i][j]);
      train_bwd_conv_layer(&net.conv3[i][j]);

      if (j == 0) {
        if (i == 0) {
          train_bwd_bn_layer(&net.conv3_branch_bn);
          train_bwd_conv_layer(&net.conv3_branch);
          train_bwd_branch_layer(&net.branch3[i]);
        }
        else {
          train_bwd_branch_layer(&net.branch3[i]);
        }
      }
    }
  }

  for (int i = B0-1; i >= 0; i--) {
    for (int j = 2; j >= 0; j--) {
      train_bwd_act_layer(&net.conv2_relu[i][j]);

      if (j == 2) {
        train_bwd_elt_layer(&net.conv2_add[i]);
      }

      train_bwd_bn_layer(&net.conv2_bn[i][j]);
      train_bwd_conv_layer(&net.conv2[i][j]);

      if (j == 0) {
        if (i == 0) {
          train_bwd_bn_layer(&net.conv2_branch_bn);
          train_bwd_conv_layer(&net.conv2_branch);
          train_bwd_branch_layer(&net.branch2[i]);
        }
        else {
          train_bwd_branch_layer(&net.branch2[i]);
        }
      }
    }
  }

  train_bwd_pool_layer(&net.pool1);
  train_bwd_act_layer(&net.conv1_relu);
  train_bwd_bn_layer(&net.conv1_bn);
  train_bwd_conv_layer(&net.conv1);
}

void resnet_connect()
{
  CONNECT(net.input, net.conv1);
  CONNECT(net.conv1, net.conv1_bn);
  CONNECT(net.conv1_bn, net.conv1_relu);
  CONNECT(net.conv1_relu, net.pool1);

  for (int i = 0; i < B0; i++) {
    if (i == 0) {
      CONNECT(net.pool1, net.branch2[i]);
      CONNECT_BRANCH_RES(net.branch2[i], net.conv2_branch, net.conv2[i][0]);
      CONNECT(net.conv2_branch, net.conv2_branch_bn);

      CONNECT(net.conv2[i][0], net.conv2_bn[i][0]);
      CONNECT(net.conv2_bn[i][0], net.conv2_relu[i][0]);
      CONNECT(net.conv2_relu[i][0], net.conv2[i][1]);
      CONNECT(net.conv2[i][1], net.conv2_bn[i][1]);
      CONNECT(net.conv2_bn[i][1], net.conv2_relu[i][1]);
      CONNECT(net.conv2_relu[i][1], net.conv2[i][2]);
      CONNECT(net.conv2[i][2], net.conv2_bn[i][2]);

      CONNECT(net.conv2_add[i], net.conv2_relu[i][2]); //Reversed order
      CONNECT_ELT(net.conv2_branch_bn, net.conv2_bn[i][2], net.conv2_add[i]);
    }
    else {
      CONNECT(net.conv2_relu[i-1][2], net.branch2[i]);
      CONNECT(net.conv2_add[i], net.conv2_relu[i][2]);
      CONNECT_DIAMOND_RES(net.branch2[i], net.conv2_add[i], net.conv2[i][0], net.conv2_bn[i][2]);

      CONNECT(net.conv2[i][0], net.conv2_bn[i][0]);
      CONNECT(net.conv2_bn[i][0], net.conv2_relu[i][0]);
      CONNECT(net.conv2_relu[i][0], net.conv2[i][1]);
      CONNECT(net.conv2[i][1], net.conv2_bn[i][1]);
      CONNECT(net.conv2_bn[i][1], net.conv2_relu[i][1]);
      CONNECT(net.conv2_relu[i][1], net.conv2[i][2]);
      CONNECT(net.conv2[i][2], net.conv2_bn[i][2]);
    }
  }

  for (int i = 0; i < B1; i++) {
    if (i == 0) {
      CONNECT(net.conv2_relu[B0-1][2], net.branch3[i]);
      CONNECT_BRANCH_RES(net.branch3[i], net.conv3_branch, net.conv3[i][0]);
      CONNECT(net.conv3_branch, net.conv3_branch_bn);

      CONNECT(net.conv3[i][0], net.conv3_bn[i][0]);
      CONNECT(net.conv3_bn[i][0], net.conv3_relu[i][0]);
      CONNECT(net.conv3_relu[i][0], net.conv3[i][1]);
      CONNECT(net.conv3[i][1], net.conv3_bn[i][1]);
      CONNECT(net.conv3_bn[i][1], net.conv3_relu[i][1]);
      CONNECT(net.conv3_relu[i][1], net.conv3[i][2]);
      CONNECT(net.conv3[i][2], net.conv3_bn[i][2]);

      CONNECT(net.conv3_add[i], net.conv3_relu[i][2]); //Reversed order
      CONNECT_ELT(net.conv3_branch_bn, net.conv3_bn[i][2], net.conv3_add[i]);
    }
    else {
      CONNECT(net.conv3_relu[i-1][2], net.branch3[i]);
      CONNECT(net.conv3_add[i], net.conv3_relu[i][2]);
      CONNECT_DIAMOND_RES(net.branch3[i], net.conv3_add[i], net.conv3[i][0], net.conv3_bn[i][2]);

      CONNECT(net.conv3[i][0], net.conv3_bn[i][0]);
      CONNECT(net.conv3_bn[i][0], net.conv3_relu[i][0]);
      CONNECT(net.conv3_relu[i][0], net.conv3[i][1]);
      CONNECT(net.conv3[i][1], net.conv3_bn[i][1]);
      CONNECT(net.conv3_bn[i][1], net.conv3_relu[i][1]);
      CONNECT(net.conv3_relu[i][1], net.conv3[i][2]);
      CONNECT(net.conv3[i][2], net.conv3_bn[i][2]);
    }
  }

  for (int i = 0; i < B2; i++) {
    if (i == 0) {
      CONNECT(net.conv3_relu[B1-1][2], net.branch4[i]);
      CONNECT_BRANCH_RES(net.branch4[i], net.conv4_branch, net.conv4[i][0]);
      CONNECT(net.conv4_branch, net.conv4_branch_bn);

      CONNECT(net.conv4[i][0], net.conv4_bn[i][0]);
      CONNECT(net.conv4_bn[i][0], net.conv4_relu[i][0]);
      CONNECT(net.conv4_relu[i][0], net.conv4[i][1]);
      CONNECT(net.conv4[i][1], net.conv4_bn[i][1]);
      CONNECT(net.conv4_bn[i][1], net.conv4_relu[i][1]);
      CONNECT(net.conv4_relu[i][1], net.conv4[i][2]);
      CONNECT(net.conv4[i][2], net.conv4_bn[i][2]);

      CONNECT(net.conv4_add[i], net.conv4_relu[i][2]); //Reversed order
      CONNECT_ELT(net.conv4_branch_bn, net.conv4_bn[i][2], net.conv4_add[i]);
    }
    else {
      CONNECT(net.conv4_relu[i-1][2], net.branch4[i]);
      CONNECT(net.conv4_add[i], net.conv4_relu[i][2]);
      CONNECT_DIAMOND_RES(net.branch4[i], net.conv4_add[i], net.conv4[i][0], net.conv4_bn[i][2]);

      CONNECT(net.conv4[i][0], net.conv4_bn[i][0]);
      CONNECT(net.conv4_bn[i][0], net.conv4_relu[i][0]);
      CONNECT(net.conv4_relu[i][0], net.conv4[i][1]);
      CONNECT(net.conv4[i][1], net.conv4_bn[i][1]);
      CONNECT(net.conv4_bn[i][1], net.conv4_relu[i][1]);
      CONNECT(net.conv4_relu[i][1], net.conv4[i][2]);
      CONNECT(net.conv4[i][2], net.conv4_bn[i][2]);
    }
  }

  for (int i = 0; i < B3; i++) {
    if (i == 0) {
      CONNECT(net.conv4_relu[B2-1][2], net.branch5[i]);
      CONNECT_BRANCH_RES(net.branch5[i], net.conv5_branch, net.conv5[i][0]);
      CONNECT(net.conv5_branch, net.conv5_branch_bn);

      CONNECT(net.conv5[i][0], net.conv5_bn[i][0]);
      CONNECT(net.conv5_bn[i][0], net.conv5_relu[i][0]);
      CONNECT(net.conv5_relu[i][0], net.conv5[i][1]);
      CONNECT(net.conv5[i][1], net.conv5_bn[i][1]);
      CONNECT(net.conv5_bn[i][1], net.conv5_relu[i][1]);
      CONNECT(net.conv5_relu[i][1], net.conv5[i][2]);
      CONNECT(net.conv5[i][2], net.conv5_bn[i][2]);

      CONNECT(net.conv5_add[i], net.conv5_relu[i][2]); //Reversed order
      CONNECT_ELT(net.conv5_branch_bn, net.conv5_bn[i][2], net.conv5_add[i]);
    }
    else {
      CONNECT(net.conv5_relu[i-1][2], net.branch5[i]);
      CONNECT(net.conv5_add[i], net.conv5_relu[i][2]);
      CONNECT_DIAMOND_RES(net.branch5[i], net.conv5_add[i], net.conv5[i][0], net.conv5_bn[i][2]);

      CONNECT(net.conv5[i][0], net.conv5_bn[i][0]);
      CONNECT(net.conv5_bn[i][0], net.conv5_relu[i][0]);
      CONNECT(net.conv5_relu[i][0], net.conv5[i][1]);
      CONNECT(net.conv5[i][1], net.conv5_bn[i][1]);
      CONNECT(net.conv5_bn[i][1], net.conv5_relu[i][1]);
      CONNECT(net.conv5_relu[i][1], net.conv5[i][2]);
      CONNECT(net.conv5[i][2], net.conv5_bn[i][2]);
    }
  }

  CONNECT(net.conv5_relu[B3-1][2], net.pool2);
  CONNECT(net.pool2, net.fc);
  CONNECT_DIRECT(net.fc, net.bias, net.softmax);
}

void resnet_clear_time()
{
  clear_time_conv_layer(&net.conv1);
  clear_time_bn_layer(&net.conv1_bn);
  clear_time_act_layer(&net.conv1_relu);
  clear_time_pool_layer(&net.pool1);

  for (int i = 0; i < B0; i++) {
    for (int j = 0; j < 3; j++) {
      if (j == 0) {
        if (i == 0) {
          clear_time_branch_layer(&net.branch2[i]);
          clear_time_conv_layer(&net.conv2_branch);
          clear_time_bn_layer(&net.conv2_branch_bn);
        }
        else {
          clear_time_branch_layer(&net.branch2[i]);
        }
      }

      clear_time_conv_layer(&net.conv2[i][j]);
      clear_time_bn_layer(&net.conv2_bn[i][j]);

      if (j == 2) {
        clear_time_elt_layer(&net.conv2_add[i]);
      }

      clear_time_act_layer(&net.conv2_relu[i][j]);
    }
  }

  for (int i = 0; i < B1; i++) {
    for (int j = 0; j < 3; j++) {
      if (j == 0) {
        if (i == 0) {
          clear_time_branch_layer(&net.branch3[i]);
          clear_time_conv_layer(&net.conv3_branch);
          clear_time_bn_layer(&net.conv3_branch_bn);
        }
        else {
          clear_time_branch_layer(&net.branch3[i]);
        }
      }

      clear_time_conv_layer(&net.conv3[i][j]);
      clear_time_bn_layer(&net.conv3_bn[i][j]);

      if (j == 2) {
        clear_time_elt_layer(&net.conv3_add[i]);
      }

      clear_time_act_layer(&net.conv3_relu[i][j]);
    }
  }

  for (int i = 0; i < B2; i++) {
    for (int j = 0; j < 3; j++) {
      if (j == 0) {
        if (i == 0) {
          clear_time_branch_layer(&net.branch4[i]);
          clear_time_conv_layer(&net.conv4_branch);
          clear_time_bn_layer(&net.conv4_branch_bn);
        }
        else {
          clear_time_branch_layer(&net.branch4[i]);
        }
      }

      clear_time_conv_layer(&net.conv4[i][j]);
      clear_time_bn_layer(&net.conv4_bn[i][j]);

      if (j == 2) {
        clear_time_elt_layer(&net.conv4_add[i]);
      }

      clear_time_act_layer(&net.conv4_relu[i][j]);
    }
  }
 
  for (int i = 0; i < B3; i++) {
    for (int j = 0; j < 3; j++) {
      if (j == 0) {
        if (i == 0) {
          clear_time_branch_layer(&net.branch5[i]);
          clear_time_conv_layer(&net.conv5_branch);
          clear_time_bn_layer(&net.conv5_branch_bn);
        }
        else {
          clear_time_branch_layer(&net.branch5[i]);
        }
      }

      clear_time_conv_layer(&net.conv5[i][j]);
      clear_time_bn_layer(&net.conv5_bn[i][j]);

      if (j == 2) {
        clear_time_elt_layer(&net.conv5_add[i]);
      }

      clear_time_act_layer(&net.conv5_relu[i][j]);
    }
  }

  clear_time_pool_layer(&net.pool2);
  clear_time_conv_layer(&net.fc);
  clear_time_bias_layer(&net.bias);
  clear_time_softmax_layer(&net.softmax);
}

void resnet_print_time()
{
  char buf[1024];
  printf("name, fwd, bwd_data, bwd_weight, update\n");

  sprintf(buf, "conv1");
  print_time_conv_layer(&net.conv1, buf);
  sprintf(buf, "conv1_bn");
  print_time_bn_layer(&net.conv1_bn, buf);
  sprintf(buf, "conv1_relu");
  print_time_act_layer(&net.conv1_relu, buf);
  sprintf(buf, "pool1");
  print_time_pool_layer(&net.pool1, buf);

  for (int i = 0; i < B0; i++) {
    for (int j = 0; j < 3; j++) {
      if (j == 0) {
        if (i == 0) {
          sprintf(buf, "branch2[%d]", i);
          print_time_branch_layer(&net.branch2[i], buf);
          sprintf(buf, "conv2_branch");
          print_time_conv_layer(&net.conv2_branch, buf);
          sprintf(buf, "conv2_branch_bn");
          print_time_bn_layer(&net.conv2_branch_bn, buf);
        }
        else {
          sprintf(buf, "branch2[%d]", i);
          print_time_branch_layer(&net.branch2[i], buf);
        }
      }

      sprintf(buf, "conv2[%d][%d]", i, j);
      print_time_conv_layer(&net.conv2[i][j], buf);
      sprintf(buf, "conv2_bn[%d][%d]", i, j);
      print_time_bn_layer(&net.conv2_bn[i][j], buf);

      if (j == 2) {
        sprintf(buf, "conv2_add[%d]", i);
        print_time_elt_layer(&net.conv2_add[i], buf);
      }

      sprintf(buf, "conv2_relu[%d][%d]", i, j);
      print_time_act_layer(&net.conv2_relu[i][j], buf);
    }
  }

  for (int i = 0; i < B1; i++) {
    for (int j = 0; j < 3; j++) {
      if (j == 0) {
        if (i == 0) {
          sprintf(buf, "branch3[%d]", i);
          print_time_branch_layer(&net.branch3[i], buf);
          sprintf(buf, "conv3_branch");
          print_time_conv_layer(&net.conv3_branch, buf);
          sprintf(buf, "conv3_branch_bn");
          print_time_bn_layer(&net.conv3_branch_bn, buf);
        }
        else {
          sprintf(buf, "branch3[%d]", i);
          print_time_branch_layer(&net.branch3[i], buf);
        }
      }

      sprintf(buf, "conv3[%d][%d]", i, j);
      print_time_conv_layer(&net.conv3[i][j], buf);
      sprintf(buf, "conv3_bn[%d][%d]", i, j);
      print_time_bn_layer(&net.conv3_bn[i][j], buf);

      if (j == 2) {
        sprintf(buf, "conv3_add[%d]", i);
        print_time_elt_layer(&net.conv3_add[i], buf);
      }

      sprintf(buf, "conv3_relu[%d][%d]", i, j);
      print_time_act_layer(&net.conv3_relu[i][j], buf);
    }
  }

  for (int i = 0; i < B2; i++) {
    for (int j = 0; j < 3; j++) {
      if (j == 0) {
        if (i == 0) {
          sprintf(buf, "branch4[%d]", i);
          print_time_branch_layer(&net.branch4[i], buf);
          sprintf(buf, "conv4_branch");
          print_time_conv_layer(&net.conv4_branch, buf);
          sprintf(buf, "conv4_branch_bn");
          print_time_bn_layer(&net.conv4_branch_bn, buf);
        }
        else {
          sprintf(buf, "branch4[%d]", i);
          print_time_branch_layer(&net.branch4[i], buf);
        }
      }

      sprintf(buf, "conv4[%d][%d]", i, j);
      print_time_conv_layer(&net.conv4[i][j], buf);
      sprintf(buf, "conv4_bn[%d][%d]", i, j);
      print_time_bn_layer(&net.conv4_bn[i][j], buf);

      if (j == 2) {
        sprintf(buf, "conv4_add[%d]", i);
        print_time_elt_layer(&net.conv4_add[i], buf);
      }

      sprintf(buf, "conv4_relu[%d][%d]", i, j);
      print_time_act_layer(&net.conv4_relu[i][j], buf);
    }
  }
 
  for (int i = 0; i < B3; i++) {
    for (int j = 0; j < 3; j++) {
      if (j == 0) {
        if (i == 0) {
          sprintf(buf, "branch5[%d]", i);
          print_time_branch_layer(&net.branch5[i], buf);
          sprintf(buf, "conv5_branch");
          print_time_conv_layer(&net.conv5_branch, buf);
          sprintf(buf, "conv5_branch_bn");
          print_time_bn_layer(&net.conv5_branch_bn, buf);
        }
        else {
          sprintf(buf, "branch5[%d]", i);
          print_time_branch_layer(&net.branch5[i], buf);
        }
      }

      sprintf(buf, "conv5[%d][%d]", i, j);
      print_time_conv_layer(&net.conv5[i][j], buf);
      sprintf(buf, "conv5_bn[%d][%d]", i, j);
      print_time_bn_layer(&net.conv5_bn[i][j], buf);

      if (j == 2) {
        sprintf(buf, "conv5_add[%d]", i);
        print_time_elt_layer(&net.conv5_add[i], buf);
      }

      sprintf(buf, "conv5_relu[%d][%d]", i, j);
      print_time_act_layer(&net.conv5_relu[i][j], buf);
    }
  }

  sprintf(buf, "pool2");
  print_time_pool_layer(&net.pool2, buf);
  sprintf(buf, "fc");
  print_time_conv_layer(&net.fc, buf);
  sprintf(buf, "bias");
  print_time_bias_layer(&net.bias, buf);
  sprintf(buf, "softmax");
  print_time_softmax_layer(&net.softmax, buf);
}

int exists(const char *fname)
{
  FILE *file;
  if ((file = fopen(fname, "r"))) {
    fclose(file);
    return 1;
  }
  return 0;
}

void verify(float *res, float *ans, int cnt)
{
  const float EPS = 1e-6;
  for (int i = 0; i < cnt; i++) {
    if (fabs(res[i]) >= EPS && fabs((res[i] - ans[i])/res[i]) >= EPS) {
      printf("%e %e relative_diff = %e\n", res[i], ans[i], fabs((res[i] - ans[i])/res[i]));
    }

    if (isnan(res[i]) || ((fabs(res[i]) >= EPS) && (fabs((res[i] - ans[i])/res[i]) >= EPS))) {
      fprintf(stderr, "Verification failed at %d, res = %lf, ans = %lf (rel diff = %lf)\n",
          i, res[i], ans[i], fabs((res[i] - ans[i])/res[i]));
      return;
    }
  }
  fprintf(stderr, "Verification success\n");
}

void cnn_train(int num_train_image, float *train_data, int *train_label) 
{
  assert(num_train_image % params.batch_size == 0); 

  resnet_init(params.batch_size);
  resnet_connect();

  int num_batches = num_train_image / params.batch_size;
  fprintf(stderr, "total iteration : %d\n", num_batches);

  size_t sz = resnet_get_param_size();
  float *param_in = (float *)malloc(sz);
  float *param_out = (float *)malloc(sz);
  float *param_result = (float *)malloc(sz);
  INITIALIZE_RAND(param_in, sz / sizeof(float));

  resnet_set_param(param_in);

  struct timespec st;
  struct timespec st_f;
  struct timespec ed;
  struct timespec ed_f;

  int first = 1;

  clock_gettime(CLOCK_MONOTONIC, &st);

  for (int e = 0; e < params.epochs; e++) {
    fprintf(stderr, "epoch %d/%d start\n", e+1, params.epochs);

    float *data_in = NULL;
    int *label_in = NULL;

    for (int b = 0; b < num_batches; b++) {
      if (first) {
        clock_gettime(CLOCK_MONOTONIC, &st_f);
      }
      int batch_size = params.batch_size;

      data_in = train_data + b * params.batch_size * params.width * params.height * params.channel;
      label_in = train_label + b * params.batch_size;

      resnet_copy_input(batch_size, data_in, label_in);

      resnet_forward();

#ifdef PRINT_LOSS
      float l = get_loss(&net.softmax, label_in);
      printf("loss for %d/%d : %f\n", b, num_batches, l);
#endif

      resnet_backward();

      if (first) {
        cudaDeviceSynchronize();
        clock_gettime(CLOCK_MONOTONIC, &ed_f);
#ifdef TIME_LAYER
        resnet_clear_time();
#endif
        first = 0;
      }
    }
  }

  cudaDeviceSynchronize();
  clock_gettime(CLOCK_MONOTONIC, &ed);          

  float training_time = diff_timespec_ms(st, ed);
  float first_training_time = diff_timespec_ms(st_f, ed_f);

  fprintf(stderr, "(Excl. 1st iter) %.3f ms, %.3f image / sec\n",
      training_time - first_training_time,
      ((float)(params.batch_size * (params.num_batch_per_epoch * params.epochs - 1)) * 1000 / (training_time - first_training_time)));
  fprintf(stderr, "(Incl. 1st iter) %.3f ms, %.3f image / sec\n",
      training_time, ((float)(params.batch_size * params.num_batch_per_epoch * params.epochs) * 1000 / (training_time)));

#ifdef TIME_LAYER
  resnet_print_time();
#endif

  resnet_get_param(param_out);

  if (exists(params.result)) {
    FILE *f = fopen(params.result, "rb");
    assert(sz  == fread(param_result, 1, sz, f));
    verify(param_out, param_result, sz / (sizeof(float))); 
    fclose(f);
  }
  else {
    FILE *f = fopen(params.result, "wb");
    fwrite(param_out, 1, sz, f);
    fclose(f);
  }
}

