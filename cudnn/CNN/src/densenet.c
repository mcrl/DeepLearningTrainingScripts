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

typedef struct densenet_s {
  input_layer input;

  conv_layer conv1;
  bn_layer bn1;
  act_layer relu1;
  pool_layer pool[5];

  bn_layer trans_bn[4];
  act_layer trans_relu[4];
  conv_layer trans_conv[3];

  branch_layer branch2[6];
  bn_layer bn2[6][2];
  act_layer relu2[6][2];
  conv_layer conv2[6][2];
  concat_layer concat2[6];

  branch_layer branch3[12];
  bn_layer bn3[12][2];
  act_layer relu3[12][2];
  conv_layer conv3[12][2];
  concat_layer concat3[12];

  branch_layer branch4[24];
  bn_layer bn4[24][2];
  act_layer relu4[24][2];
  conv_layer conv4[24][2];
  concat_layer concat4[24];

  branch_layer branch5[16];
  bn_layer bn5[16][2];
  act_layer relu5[16][2];
  conv_layer conv5[16][2];
  concat_layer concat5[16];

  conv_layer fc;
  bias_layer bias;
  softmax_layer softmax;
} densenet;

densenet net;

bool is_initiated = false;

void params_modify()
{
}

void densenet_init(int batch_size)
{
  chkCUDNN(cudnnCreate(&cudnn));
  chkCUBLAS(cublasCreate(&cublas));
  srand(params.seed);

  init_input_layer(&net.input, cudnn, batch_size, 3, 224, 224);
  init_conv_layer(&net.conv1, cudnn, batch_size, 7, 7, 3, 3, 2, 2, 3, 64, 224, 224);
  init_bn_layer(&net.bn1, cudnn, batch_size, 64, 112, 112); 
  init_act_layer(&net.relu1, cudnn, batch_size, 64, 112, 112);
  init_pool_layer(&net.pool[0], cudnn, batch_size, 3, 3, 1, 1, 2, 2, 64, 112, 112, max);

  int ch_in[2];
  for (int i = 0; i < 6; i++) {
    init_branch_layer(&net.branch2[i], cudnn, batch_size, 2, 64 + 32 * i, 56, 56);
    init_bn_layer(&net.bn2[i][0], cudnn, batch_size, 64 + 32 * i, 56, 56);
    init_act_layer(&net.relu2[i][0], cudnn, batch_size, 64 + 32 * i, 56, 56);
    init_conv_layer(&net.conv2[i][0], cudnn, batch_size, 1, 1, 0, 0, 1, 1, 64 + 32 * i, 128, 56, 56);
    init_bn_layer(&net.bn2[i][1], cudnn, batch_size, 128, 56, 56);
    init_act_layer(&net.relu2[i][1], cudnn, batch_size, 128, 56, 56);
    init_conv_layer(&net.conv2[i][1], cudnn, batch_size, 3, 3, 1, 1, 1, 1, 128, 32, 56, 56);
    ch_in[1] = 32;
    ch_in[0] = 64 + 32*i;
    init_concat_layer(&net.concat2[i], cudnn, batch_size, 2, ch_in, 56, 56);
  }

  init_bn_layer(&net.trans_bn[0], cudnn, batch_size, 256, 56, 56); 
  init_act_layer(&net.trans_relu[0], cudnn, batch_size, 256, 56, 56);
  init_conv_layer(&net.trans_conv[0], cudnn, batch_size, 1, 1, 0, 0, 1, 1, 256, 128, 56, 56);
  init_pool_layer(&net.pool[1], cudnn, batch_size, 2, 2, 0, 0, 2, 2, 128, 56, 56, max);

  for (int i = 0; i < 12; i++) {
    init_branch_layer(&net.branch3[i], cudnn, batch_size, 2, 128 + 32 * i, 28, 28);
    init_bn_layer(&net.bn3[i][0], cudnn, batch_size, 128 + 32 * i, 28, 28);
    init_act_layer(&net.relu3[i][0], cudnn, batch_size, 128 + 32 * i, 28, 28);
    init_conv_layer(&net.conv3[i][0], cudnn, batch_size, 1, 1, 0, 0, 1, 1, 128 + 32 * i, 128, 28, 28);
    init_bn_layer(&net.bn3[i][1], cudnn, batch_size, 128, 28, 28);
    init_act_layer(&net.relu3[i][1], cudnn, batch_size, 128, 28, 28);
    init_conv_layer(&net.conv3[i][1], cudnn, batch_size, 3, 3, 1, 1, 1, 1, 128, 32, 28, 28);
    ch_in[1] = 32;
    ch_in[0] = 128 + 32*i;
    init_concat_layer(&net.concat3[i], cudnn, batch_size, 2, ch_in, 28, 28);
  }

  init_bn_layer(&net.trans_bn[1], cudnn, batch_size, 512, 28, 28); 
  init_act_layer(&net.trans_relu[1], cudnn, batch_size, 512, 28, 28);
  init_conv_layer(&net.trans_conv[1], cudnn, batch_size, 1, 1, 0, 0, 1, 1, 512, 256, 28, 28);
  init_pool_layer(&net.pool[2], cudnn, batch_size, 2, 2, 0, 0, 2, 2, 256, 28, 28, max);

  for (int i = 0; i < 24; i++) {
    init_branch_layer(&net.branch4[i], cudnn, batch_size, 2, 256 + 32 * i, 14, 14);
    init_bn_layer(&net.bn4[i][0], cudnn, batch_size, 256 + 32 * i, 14, 14);
    init_act_layer(&net.relu4[i][0], cudnn, batch_size, 256 + 32 * i, 14, 14);
    init_conv_layer(&net.conv4[i][0], cudnn, batch_size, 1, 1, 0, 0, 1, 1, 256 + 32 * i, 256, 14, 14);
    init_bn_layer(&net.bn4[i][1], cudnn, batch_size, 256, 14, 14);
    init_act_layer(&net.relu4[i][1], cudnn, batch_size, 256, 14, 14);
    init_conv_layer(&net.conv4[i][1], cudnn, batch_size, 3, 3, 1, 1, 1, 1, 256, 32, 14, 14);
    ch_in[1] = 32;
    ch_in[0] = 256 + 32*i;
    init_concat_layer(&net.concat4[i], cudnn, batch_size, 2, ch_in, 14, 14);
  }

  init_bn_layer(&net.trans_bn[2], cudnn, batch_size, 1024, 14, 14); 
  init_act_layer(&net.trans_relu[2], cudnn, batch_size, 1024, 14, 14);
  init_conv_layer(&net.trans_conv[2], cudnn, batch_size, 1, 1, 0, 0, 1, 1, 1024, 512, 14, 14);
  init_pool_layer(&net.pool[3], cudnn, batch_size, 2, 2, 0, 0, 2, 2, 512, 14, 14, max);

  for (int i = 0; i < 16; i++) {
    init_branch_layer(&net.branch5[i], cudnn, batch_size, 2, 512 + 32 * i, 7, 7);
    init_bn_layer(&net.bn5[i][0], cudnn, batch_size, 512 + 32 * i, 7, 7);
    init_act_layer(&net.relu5[i][0], cudnn, batch_size, 512 + 32 * i, 7, 7);
    init_conv_layer(&net.conv5[i][0], cudnn, batch_size, 1, 1, 0, 0, 1, 1, 512 + 32 * i, 512, 7, 7);
    init_bn_layer(&net.bn5[i][1], cudnn, batch_size, 512, 7, 7);
    init_act_layer(&net.relu5[i][1], cudnn, batch_size, 512, 7, 7);
    init_conv_layer(&net.conv5[i][1], cudnn, batch_size, 3, 3, 1, 1, 1, 1, 512, 32, 7, 7);
    ch_in[1] = 32;
    ch_in[0] = 512 + 32*i;
    init_concat_layer(&net.concat5[i], cudnn, batch_size, 2, ch_in, 7, 7);
  }

  init_bn_layer(&net.trans_bn[3], cudnn, batch_size, 1024, 7, 7); 
  init_act_layer(&net.trans_relu[3], cudnn, batch_size, 1024, 7, 7);
  init_pool_layer(&net.pool[4], cudnn, batch_size, 7, 7, 0, 0, 1, 1, 1024, 7, 7, max);

  init_conv_layer(&net.fc, cudnn, batch_size, 1, 1, 0, 0, 1, 1, 1024, 1000, 1, 1);
  init_bias_layer(&net.bias, cudnn, batch_size, 1000, 1, 1);
  init_softmax_layer(&net.softmax, cudnn, batch_size, 1000);

  init_conv_workspace();

  is_initiated = true;
}

size_t densenet_get_param_size()
{
  size_t sum = 0;

  sum += PSIZE_CONV(net.conv1);
  sum += PSIZE_BN(net.bn1);

  for (int i = 0; i < 3; i++) {
    sum += PSIZE_BN(net.trans_bn[i]);
    sum += PSIZE_CONV(net.trans_conv[i]);
  }

  for (int i = 0; i < 6; i++)
    for (int j = 0; j < 2; j++) {
      sum += PSIZE_BN(net.bn2[i][j]);
      sum += PSIZE_CONV(net.conv2[i][j]);
    }

  for (int i = 0; i < 12; i++)
    for (int j = 0; j < 2; j++) {
      sum += PSIZE_BN(net.bn3[i][j]);
      sum += PSIZE_CONV(net.conv3[i][j]);
    }

  for (int i = 0; i < 24; i++)
    for (int j = 0; j < 2; j++) {
      sum += PSIZE_BN(net.bn4[i][j]);
      sum += PSIZE_CONV(net.conv4[i][j]);
    }

  for (int i = 0; i < 16; i++)
    for (int j = 0; j < 2; j++) {
      sum += PSIZE_BN(net.bn5[i][j]);
      sum += PSIZE_CONV(net.conv5[i][j]);
    }
  
  sum += PSIZE_CONV(net.fc);
  sum += PSIZE_BIAS(net.bias);

  return sum;
}

void densenet_set_param(float *param)
{
  INIT_CONV(net.conv1); 
  INIT_BN(net.bn1);

  for (int i = 0; i < 3; i++) {
    INIT_BN(net.trans_bn[i]);
    INIT_CONV(net.trans_conv[i]);
  }

  for (int i = 0; i < 6; i++)
    for (int j = 0; j < 2; j++) {
       INIT_BN(net.bn2[i][j]);
       INIT_CONV(net.conv2[i][j]);
    }

  for (int i = 0; i < 12; i++)
    for (int j = 0; j < 2; j++) {
       INIT_BN(net.bn3[i][j]);
       INIT_CONV(net.conv3[i][j]);
    }

  for (int i = 0; i < 24; i++)
    for (int j = 0; j < 2; j++) {
       INIT_BN(net.bn4[i][j]);
       INIT_CONV(net.conv4[i][j]);
    }

  for (int i = 0; i < 16; i++)
    for (int j = 0; j < 2; j++) {
       INIT_BN(net.bn5[i][j]);
       INIT_CONV(net.conv5[i][j]);
    }
  
  INIT_CONV(net.fc);
  INIT_BIAS(net.bias);
}

void densenet_get_param(float *param)
{
  GET_CONV(net.conv1); 
  GET_BN(net.bn1);

  for (int i = 0; i < 3; i++) {
    GET_BN(net.trans_bn[i]);
    GET_CONV(net.trans_conv[i]);
  }

  GET_BN(net.trans_bn[3]);

  for (int i = 0; i < 6; i++)
    for (int j = 0; j < 2; j++) {
       GET_BN(net.bn2[i][j]);
       GET_CONV(net.conv2[i][j]);
    }

  for (int i = 0; i < 12; i++)
    for (int j = 0; j < 2; j++) {
       GET_BN(net.bn3[i][j]);
       GET_CONV(net.conv3[i][j]);
    }

  for (int i = 0; i < 24; i++)
    for (int j = 0; j < 2; j++) {
       GET_BN(net.bn4[i][j]);
       GET_CONV(net.conv4[i][j]);
    }

  for (int i = 0; i < 16; i++)
    for (int j = 0; j < 2; j++) {
       GET_BN(net.bn5[i][j]);
       GET_CONV(net.conv5[i][j]);
    }
  
  GET_CONV(net.fc);
  GET_BIAS(net.bias);
}

void densenet_copy_input(int batch_size, float * data_in, int * label_in)
{
  size_t input_size = sizeof(float) * batch_size * params.width * params.height * params.channel;

  chkCUDA(cudaMemcpy(net.input.output, data_in, input_size, cudaMemcpyHostToDevice)); 
  chkCUDA(cudaMemcpy(net.softmax.label_in, label_in, batch_size * sizeof(int), cudaMemcpyHostToDevice)); 
  cuda_set_label(batch_size, 1000, net.softmax.label_in, net.softmax.label);
}

void densenet_forward()
{
  train_fwd_conv_layer(&net.conv1);
  train_fwd_bn_layer(&net.bn1);
  train_fwd_act_layer(&net.relu1);
  train_fwd_pool_layer(&net.pool[0]);

  for (int i = 0; i < 6; i++) {
    train_fwd_branch_layer(&net.branch2[i]);
    train_fwd_bn_layer(&net.bn2[i][0]);
    train_fwd_act_layer(&net.relu2[i][0]);
    train_fwd_conv_layer(&net.conv2[i][0]);
    train_fwd_bn_layer(&net.bn2[i][1]);
    train_fwd_act_layer(&net.relu2[i][1]);
    train_fwd_conv_layer(&net.conv2[i][1]);
    train_fwd_concat_layer(&net.concat2[i]);
  }

  train_fwd_bn_layer(&net.trans_bn[0]);
  train_fwd_act_layer(&net.trans_relu[0]);
  train_fwd_conv_layer(&net.trans_conv[0]);
  train_fwd_pool_layer(&net.pool[1]);

  for (int i = 0; i < 12; i++) {
    train_fwd_branch_layer(&net.branch3[i]);
    train_fwd_bn_layer(&net.bn3[i][0]);
    train_fwd_act_layer(&net.relu3[i][0]);
    train_fwd_conv_layer(&net.conv3[i][0]);
    train_fwd_bn_layer(&net.bn3[i][1]);
    train_fwd_act_layer(&net.relu3[i][1]);
    train_fwd_conv_layer(&net.conv3[i][1]);
    train_fwd_concat_layer(&net.concat3[i]);
  }

  train_fwd_bn_layer(&net.trans_bn[1]);
  train_fwd_act_layer(&net.trans_relu[1]);
  train_fwd_conv_layer(&net.trans_conv[1]);
  train_fwd_pool_layer(&net.pool[2]);
  
  for (int i = 0; i < 24; i++) {
    train_fwd_branch_layer(&net.branch4[i]);
    train_fwd_bn_layer(&net.bn4[i][0]);
    train_fwd_act_layer(&net.relu4[i][0]);
    train_fwd_conv_layer(&net.conv4[i][0]);
    train_fwd_bn_layer(&net.bn4[i][1]);
    train_fwd_act_layer(&net.relu4[i][1]);
    train_fwd_conv_layer(&net.conv4[i][1]);
    train_fwd_concat_layer(&net.concat4[i]);
  }

  train_fwd_bn_layer(&net.trans_bn[2]);
  train_fwd_act_layer(&net.trans_relu[2]);
  train_fwd_conv_layer(&net.trans_conv[2]);
  train_fwd_pool_layer(&net.pool[3]);

  for (int i = 0; i < 16; i++) {
    train_fwd_branch_layer(&net.branch5[i]);
    train_fwd_bn_layer(&net.bn5[i][0]);
    train_fwd_act_layer(&net.relu5[i][0]);
    train_fwd_conv_layer(&net.conv5[i][0]);
    train_fwd_bn_layer(&net.bn5[i][1]);
    train_fwd_act_layer(&net.relu5[i][1]);
    train_fwd_conv_layer(&net.conv5[i][1]);
    train_fwd_concat_layer(&net.concat5[i]);
  }

  train_fwd_bn_layer(&net.trans_bn[3]);
  train_fwd_act_layer(&net.trans_relu[3]);
  train_fwd_pool_layer(&net.pool[4]);

  train_fwd_conv_layer(&net.fc);
  train_fwd_bias_layer(&net.bias);
  train_fwd_softmax_layer(&net.softmax);
}

void densenet_backward()
{
  train_bwd_softmax_layer(&net.softmax);
  train_bwd_bias_layer(&net.bias);
  train_bwd_conv_layer(&net.fc);

  train_bwd_pool_layer(&net.pool[4]);
  train_bwd_act_layer(&net.trans_relu[3]);
  train_bwd_bn_layer(&net.trans_bn[3]);

  for (int i = 15; i >= 0; i--) {
    train_bwd_concat_layer(&net.concat5[i]);
    train_bwd_conv_layer(&net.conv5[i][1]);
    train_bwd_act_layer(&net.relu5[i][1]);
    train_bwd_bn_layer(&net.bn5[i][1]);
    train_bwd_conv_layer(&net.conv5[i][0]);
    train_bwd_act_layer(&net.relu5[i][0]);
    train_bwd_bn_layer(&net.bn5[i][0]);
    train_bwd_branch_layer(&net.branch5[i]);
  }

  train_bwd_pool_layer(&net.pool[3]);
  train_bwd_conv_layer(&net.trans_conv[2]);
  train_bwd_act_layer(&net.trans_relu[2]);
  train_bwd_bn_layer(&net.trans_bn[2]);

  for (int i = 23; i >= 0; i--)
  {
    train_bwd_concat_layer(&net.concat4[i]);
    train_bwd_conv_layer(&net.conv4[i][1]);
    train_bwd_act_layer(&net.relu4[i][1]);
    train_bwd_bn_layer(&net.bn4[i][1]);
    train_bwd_conv_layer(&net.conv4[i][0]);
    train_bwd_act_layer(&net.relu4[i][0]);
    train_bwd_bn_layer(&net.bn4[i][0]);
    train_bwd_branch_layer(&net.branch4[i]);
  }

  train_bwd_pool_layer(&net.pool[2]);
  train_bwd_conv_layer(&net.trans_conv[1]);
  train_bwd_act_layer(&net.trans_relu[1]);
  train_bwd_bn_layer(&net.trans_bn[1]);

  for (int i = 11; i >= 0; i--) {
    train_bwd_concat_layer(&net.concat3[i]);
    train_bwd_conv_layer(&net.conv3[i][1]);
    train_bwd_act_layer(&net.relu3[i][1]);
    train_bwd_bn_layer(&net.bn3[i][1]);
    train_bwd_conv_layer(&net.conv3[i][0]);
    train_bwd_act_layer(&net.relu3[i][0]);
    train_bwd_bn_layer(&net.bn3[i][0]);
    train_bwd_branch_layer(&net.branch3[i]);
  }

  train_bwd_pool_layer(&net.pool[1]);
  train_bwd_conv_layer(&net.trans_conv[0]);
  train_bwd_act_layer(&net.trans_relu[0]);
  train_bwd_bn_layer(&net.trans_bn[0]);

  for (int i = 5; i >= 0; i--) {
    train_bwd_concat_layer(&net.concat2[i]);
    train_bwd_conv_layer(&net.conv2[i][1]);
    train_bwd_act_layer(&net.relu2[i][1]);
    train_bwd_bn_layer(&net.bn2[i][1]);
    train_bwd_conv_layer(&net.conv2[i][0]);
    train_bwd_act_layer(&net.relu2[i][0]);
    train_bwd_bn_layer(&net.bn2[i][0]);
    train_bwd_branch_layer(&net.branch2[i]);
  }

  train_bwd_pool_layer(&net.pool[0]);
  train_bwd_act_layer(&net.relu1);
  train_bwd_bn_layer(&net.bn1);
  train_bwd_conv_layer(&net.conv1);
}

void densenet_connect()
{
  CONNECT(net.input, net.conv1);
  CONNECT(net.conv1, net.bn1);
  CONNECT(net.bn1, net.relu1);
  CONNECT(net.relu1, net.pool[0]);

  for (int i = 0; i < 6; i++) {
    if (i == 0) {
      CONNECT(net.pool[0], net.branch2[i]);
    }
    else {
      CONNECT(net.concat2[i-1], net.branch2[i]);
    }
    CONNECT_DIAMOND_DENSE(net.branch2[i], net.concat2[i], net.bn2[i][0], net.conv2[i][1]);
    CONNECT(net.bn2[i][0], net.relu2[i][0]);
    CONNECT(net.relu2[i][0], net.conv2[i][0]);
    CONNECT(net.conv2[i][0], net.bn2[i][1]);
    CONNECT(net.bn2[i][1], net.relu2[i][1]);
    CONNECT(net.relu2[i][1], net.conv2[i][1]);
  }

  CONNECT(net.concat2[5], net.trans_bn[0]);
  CONNECT(net.trans_bn[0], net.trans_relu[0]);
  CONNECT(net.trans_relu[0], net.trans_conv[0]);
  CONNECT(net.trans_conv[0], net.pool[1]);

  for (int i = 0; i < 12; i++) {
    if (i == 0) {
      CONNECT(net.pool[1], net.branch3[i]);
    }
    else {
      CONNECT(net.concat3[i-1], net.branch3[i]);
    }
    CONNECT_DIAMOND_DENSE(net.branch3[i], net.concat3[i], net.bn3[i][0], net.conv3[i][1]);
    CONNECT(net.bn3[i][0], net.relu3[i][0]);
    CONNECT(net.relu3[i][0], net.conv3[i][0]);
    CONNECT(net.conv3[i][0], net.bn3[i][1]);
    CONNECT(net.bn3[i][1], net.relu3[i][1]);
    CONNECT(net.relu3[i][1], net.conv3[i][1]);
  }

  CONNECT(net.concat3[11], net.trans_bn[1]);
  CONNECT(net.trans_bn[1], net.trans_relu[1]);
  CONNECT(net.trans_relu[1], net.trans_conv[1]);
  CONNECT(net.trans_conv[1], net.pool[2]);

  for (int i = 0; i < 24; i++) {
    if (i == 0) {
      CONNECT(net.pool[2], net.branch4[i]);
    }
    else {
      CONNECT(net.concat4[i-1], net.branch4[i]);
    }
    CONNECT_DIAMOND_DENSE(net.branch4[i], net.concat4[i], net.bn4[i][0], net.conv4[i][1]);
    CONNECT(net.bn4[i][0], net.relu4[i][0]);
    CONNECT(net.relu4[i][0], net.conv4[i][0]);
    CONNECT(net.conv4[i][0], net.bn4[i][1]);
    CONNECT(net.bn4[i][1], net.relu4[i][1]);
    CONNECT(net.relu4[i][1], net.conv4[i][1]);
  }

  CONNECT(net.concat4[23], net.trans_bn[2]);
  CONNECT(net.trans_bn[2], net.trans_relu[2]);
  CONNECT(net.trans_relu[2], net.trans_conv[2]);
  CONNECT(net.trans_conv[2], net.pool[3]);

  for (int i = 0; i < 16; i++) {
    if (i == 0) {
      CONNECT(net.pool[3], net.branch5[i]);
    }
    else {
      CONNECT(net.concat5[i-1], net.branch5[i]);
    }
    CONNECT_DIAMOND_DENSE(net.branch5[i], net.concat5[i], net.bn5[i][0], net.conv5[i][1]);
    CONNECT(net.bn5[i][0], net.relu5[i][0]);
    CONNECT(net.relu5[i][0], net.conv5[i][0]);
    CONNECT(net.conv5[i][0], net.bn5[i][1]);
    CONNECT(net.bn5[i][1], net.relu5[i][1]);
    CONNECT(net.relu5[i][1], net.conv5[i][1]);
  }

  CONNECT(net.concat5[15], net.trans_bn[3]);
  CONNECT(net.trans_bn[3], net.trans_relu[3]);
  CONNECT(net.trans_relu[3], net.pool[4]);
  CONNECT(net.pool[4], net.fc);
  CONNECT_DIRECT(net.fc, net.bias, net.softmax);
}

void densenet_clear_time()
{
  clear_time_conv_layer(&net.conv1);
  clear_time_bn_layer(&net.bn1);
  clear_time_act_layer(&net.relu1);
  clear_time_pool_layer(&net.pool[0]);

  for (int i = 0; i < 6; i++) {
    clear_time_branch_layer(&net.branch2[i]);
    clear_time_bn_layer(&net.bn2[i][0]);
    clear_time_act_layer(&net.relu2[i][0]);
    clear_time_conv_layer(&net.conv2[i][0]);
    clear_time_bn_layer(&net.bn2[i][1]);
    clear_time_act_layer(&net.relu2[i][1]);
    clear_time_conv_layer(&net.conv2[i][1]);
    clear_time_concat_layer(&net.concat2[i]);
  }

  clear_time_bn_layer(&net.trans_bn[0]);
  clear_time_act_layer(&net.trans_relu[0]);
  clear_time_conv_layer(&net.trans_conv[0]);
  clear_time_pool_layer(&net.pool[1]);

  for (int i = 0; i < 12; i++) {
    clear_time_branch_layer(&net.branch3[i]);
    clear_time_bn_layer(&net.bn3[i][0]);
    clear_time_act_layer(&net.relu3[i][0]);
    clear_time_conv_layer(&net.conv3[i][0]);
    clear_time_bn_layer(&net.bn3[i][1]);
    clear_time_act_layer(&net.relu3[i][1]);
    clear_time_conv_layer(&net.conv3[i][1]);
    clear_time_concat_layer(&net.concat3[i]);
  }

  clear_time_bn_layer(&net.trans_bn[1]);
  clear_time_act_layer(&net.trans_relu[1]);
  clear_time_conv_layer(&net.trans_conv[1]);
  clear_time_pool_layer(&net.pool[2]);

  for (int i = 0; i < 24; i++) {
    clear_time_branch_layer(&net.branch4[i]);
    clear_time_bn_layer(&net.bn4[i][0]);
    clear_time_act_layer(&net.relu4[i][0]);
    clear_time_conv_layer(&net.conv4[i][0]);
    clear_time_bn_layer(&net.bn4[i][1]);
    clear_time_act_layer(&net.relu4[i][1]);
    clear_time_conv_layer(&net.conv4[i][1]);
    clear_time_concat_layer(&net.concat4[i]);
  }

  clear_time_bn_layer(&net.trans_bn[2]);
  clear_time_act_layer(&net.trans_relu[2]);
  clear_time_conv_layer(&net.trans_conv[2]);
  clear_time_pool_layer(&net.pool[3]);

  for (int i = 0; i < 16; i++) {
    clear_time_branch_layer(&net.branch5[i]);
    clear_time_bn_layer(&net.bn5[i][0]);
    clear_time_act_layer(&net.relu5[i][0]);
    clear_time_conv_layer(&net.conv5[i][0]);
    clear_time_bn_layer(&net.bn5[i][1]);
    clear_time_act_layer(&net.relu5[i][1]);
    clear_time_conv_layer(&net.conv5[i][1]);
    clear_time_concat_layer(&net.concat5[i]);
  }

  clear_time_bn_layer(&net.trans_bn[3]);
  clear_time_act_layer(&net.trans_relu[3]);
  clear_time_pool_layer(&net.pool[4]);

  clear_time_conv_layer(&net.fc);
  clear_time_bias_layer(&net.bias);
  clear_time_softmax_layer(&net.softmax);
}

void densenet_print_time()
{
  char buf[1024];
  printf("name, fwd, bwd_data, bwd_weight, update\n");

  print_time_conv_layer(&net.conv1, "conv1");
  print_time_bn_layer(&net.bn1, "bn1");
  print_time_act_layer(&net.relu1, "relu1");
  print_time_pool_layer(&net.pool[0], "pool0");

  for (int i = 0; i < 6; i++) {
    sprintf(buf, "branch2[%d]", i);
    print_time_branch_layer(&net.branch2[i], buf);
    sprintf(buf, "bn2[%d][0]", i);
    print_time_bn_layer(&net.bn2[i][0], buf);
    sprintf(buf, "relu2[%d][0]", i);
    print_time_act_layer(&net.relu2[i][0], buf);
    sprintf(buf, "conv2[%d][0]", i);
    print_time_conv_layer(&net.conv2[i][0], buf);
    sprintf(buf, "bn2[%d][1]", i);
    print_time_bn_layer(&net.bn2[i][1], buf);
    sprintf(buf, "relu2[%d][1]", i);
    print_time_act_layer(&net.relu2[i][1], buf);
    sprintf(buf, "conv2[%d][1]", i);
    print_time_conv_layer(&net.conv2[i][1], buf);
    sprintf(buf, "concat2[%d]", i);
    print_time_concat_layer(&net.concat2[i], buf);
  }

  sprintf(buf, "trans_bn[0]");
  print_time_bn_layer(&net.trans_bn[0], buf);
  sprintf(buf, "trans_relu[0]");
  print_time_act_layer(&net.trans_relu[0], buf);
  sprintf(buf, "trans_conv[0]");
  print_time_conv_layer(&net.trans_conv[0], buf);
  sprintf(buf, "pool[1]");
  print_time_pool_layer(&net.pool[1], buf);

  for (int i = 0; i < 12; i++) {
    sprintf(buf, "branch3[%d]", i);
    print_time_branch_layer(&net.branch3[i], buf);
    sprintf(buf, "bn3[%d][0]", i);
    print_time_bn_layer(&net.bn3[i][0], buf);
    sprintf(buf, "relu3[%d][0]", i);
    print_time_act_layer(&net.relu3[i][0], buf);
    sprintf(buf, "conv3[%d][0]", i);
    print_time_conv_layer(&net.conv3[i][0], buf);
    sprintf(buf, "bn3[%d][1]", i);
    print_time_bn_layer(&net.bn3[i][1], buf);
    sprintf(buf, "relu3[%d][1]", i);
    print_time_act_layer(&net.relu3[i][1], buf);
    sprintf(buf, "conv3[%d][1]", i);
    print_time_conv_layer(&net.conv3[i][1], buf);
    sprintf(buf, "concat3[%d]", i);
    print_time_concat_layer(&net.concat3[i], buf);
  }

  sprintf(buf, "trans_bn[1]");
  print_time_bn_layer(&net.trans_bn[1], buf);
  sprintf(buf, "trans_relu[1]");
  print_time_act_layer(&net.trans_relu[1], buf);
  sprintf(buf, "trans_conv[1]");
  print_time_conv_layer(&net.trans_conv[1], buf);
  sprintf(buf, "pool[2]");
  print_time_pool_layer(&net.pool[2], buf);

  for (int i = 0; i < 24; i++) {
    sprintf(buf, "branch4[%d]", i);
    print_time_branch_layer(&net.branch4[i], buf);
    sprintf(buf, "bn4[%d][0]", i);
    print_time_bn_layer(&net.bn4[i][0], buf);
    sprintf(buf, "relu4[%d][0]", i);
    print_time_act_layer(&net.relu4[i][0], buf);
    sprintf(buf, "conv4[%d][0]", i);
    print_time_conv_layer(&net.conv4[i][0], buf);
    sprintf(buf, "bn4[%d][1]", i);
    print_time_bn_layer(&net.bn4[i][1], buf);
    sprintf(buf, "relu4[%d][1]", i);
    print_time_act_layer(&net.relu4[i][1], buf);
    sprintf(buf, "conv4[%d][1]", i);
    print_time_conv_layer(&net.conv4[i][1], buf);
    sprintf(buf, "concat4[%d]", i);
    print_time_concat_layer(&net.concat4[i], buf);
  }

  sprintf(buf, "trans_bn[2]");
  print_time_bn_layer(&net.trans_bn[2], buf);
  sprintf(buf, "trans_relu[2]");
  print_time_act_layer(&net.trans_relu[2], buf);
  sprintf(buf, "trans_conv[2]");
  print_time_conv_layer(&net.trans_conv[2], buf);
  sprintf(buf, "pool[3]");
  print_time_pool_layer(&net.pool[3], buf);

  for (int i = 0; i < 16; i++) {
    sprintf(buf, "branch5[%d]", i);
    print_time_branch_layer(&net.branch5[i], buf);
    sprintf(buf, "bn5[%d][0]", i);
    print_time_bn_layer(&net.bn5[i][0], buf);
    sprintf(buf, "relu5[%d][0]", i);
    print_time_act_layer(&net.relu5[i][0], buf);
    sprintf(buf, "conv5[%d][0]", i);
    print_time_conv_layer(&net.conv5[i][0], buf);
    sprintf(buf, "bn5[%d][1]", i);
    print_time_bn_layer(&net.bn5[i][1], buf);
    sprintf(buf, "relu5[%d][1]", i);
    print_time_act_layer(&net.relu5[i][1], buf);
    sprintf(buf, "conv5[%d][1]", i);
    print_time_conv_layer(&net.conv5[i][1], buf);
    sprintf(buf, "concat5[%d]", i);
    print_time_concat_layer(&net.concat5[i], buf);
  }

  sprintf(buf, "trans_bn[3]");
  print_time_bn_layer(&net.trans_bn[3], buf);
  sprintf(buf, "trans_relu[3]");
  print_time_act_layer(&net.trans_relu[3], buf);
  sprintf(buf, "pool[4]");
  print_time_pool_layer(&net.pool[4], buf);

  print_time_conv_layer(&net.fc, "fc");
  print_time_bias_layer(&net.bias, "bias");
  print_time_softmax_layer(&net.softmax, "softmax");
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

  densenet_init(params.batch_size);
  densenet_connect();

  int num_batches = num_train_image / params.batch_size;
  fprintf(stderr, "total iteration : %d\n", num_batches);

  size_t sz = densenet_get_param_size();
  float *param_in = (float *)malloc(sz);
  float *param_out = (float *)malloc(sz);
  float *param_result = (float *)malloc(sz);
  INITIALIZE_RAND(param_in, sz/sizeof(float));

  densenet_set_param(param_in);

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

      densenet_copy_input(batch_size, data_in, label_in);

      densenet_forward();

#ifdef PRINT_LOSS
      float l = get_loss(&net.softmax, label_in);
      printf("loss for %d/%d : %f\n", b, num_batches, l);
#endif

      densenet_backward();

      if (first) {
        cudaDeviceSynchronize();
        clock_gettime(CLOCK_MONOTONIC, &ed_f);
#ifdef TIME_LAYER
        densenet_clear_time();
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
  densenet_print_time();
#endif

  densenet_get_param(param_out);

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

