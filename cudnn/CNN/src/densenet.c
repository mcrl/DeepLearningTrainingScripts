#include <stdio.h>
#include <string.h>
#include <stdbool.h>
#include <assert.h>

#include "cnn.h"
#include "params.h"
#include "layer.h"
#include "utils.h"
#include "execute.h"

/* MPI */
extern int node_id;

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

  bool is_initiated;
} densenet;

densenet net = {
  .is_initiated = false
};

void params_modify()
{
}

void densenet_init(int batch_size)
{
  char name[1024];

  srand(params.seed);

  sprintf(name, "input");
  init_input_layer(&net.input, name, batch_size, 3, 224, 224);

  sprintf(name, "conv1");
  init_conv_layer(&net.conv1, name, batch_size, 7, 7, 3, 3, 2, 2, 3, 64, 224, 224);

  sprintf(name, "bn1");
  init_bn_layer(&net.bn1, name, batch_size, 64, 112, 112); 

  sprintf(name, "relu1");
  init_act_layer(&net.relu1, name, batch_size, 64, 112, 112, RELU_T);

  sprintf(name, "pool[0]");
  init_pool_layer(&net.pool[0], name, batch_size, 3, 3, 1, 1, 2, 2, 64, 112, 112, MAX_T);

  int ch_in[2];
  for (int i = 0; i < 6; i++) {
    ch_in[0] = 64 + 32 * i;
    ch_in[1] = 32;

    sprintf(name, "branch2[%d]", i);
    init_branch_layer(&net.branch2[i], name, batch_size, 2, ch_in[0], 56, 56);

    sprintf(name, "bn2[%d][0]", i);
    init_bn_layer(&net.bn2[i][0], name, batch_size, ch_in[0], 56, 56);

    sprintf(name, "relu2[%d][0]", i);
    init_act_layer(&net.relu2[i][0], name, batch_size, ch_in[0], 56, 56, RELU_T);

    sprintf(name, "conv2[%d][0]", i);
    init_conv_layer(&net.conv2[i][0], name, batch_size, 1, 1, 0, 0, 1, 1, ch_in[0], 128, 56, 56);

    sprintf(name, "bn2[%d][1]", i);
    init_bn_layer(&net.bn2[i][1], name, batch_size, 128, 56, 56);

    sprintf(name, "relu2[%d][1]", i);
    init_act_layer(&net.relu2[i][1], name, batch_size, 128, 56, 56, RELU_T);

    sprintf(name, "conv2[%d][1]", i);
    init_conv_layer(&net.conv2[i][1], name, batch_size, 3, 3, 1, 1, 1, 1, 128, 32, 56, 56);

    sprintf(name, "concat2[%d]", i);
    init_concat_layer(&net.concat2[i], name, batch_size, 2, ch_in, 56, 56);
  }

  sprintf(name, "trans_bn[0]");
  init_bn_layer(&net.trans_bn[0], name, batch_size, 256, 56, 56); 

  sprintf(name, "trans_relu[0]");
  init_act_layer(&net.trans_relu[0], name, batch_size, 256, 56, 56, RELU_T);

  sprintf(name, "trans_conv[0]");
  init_conv_layer(&net.trans_conv[0], name, batch_size, 1, 1, 0, 0, 1, 1, 256, 128, 56, 56);

  sprintf(name, "pool[1]");
  init_pool_layer(&net.pool[1], name, batch_size, 2, 2, 0, 0, 2, 2, 128, 56, 56, MAX_T);

  for (int i = 0; i < 12; i++) {
    ch_in[0] = 128 + 32 * i;
    ch_in[1] = 32;

    sprintf(name, "branch3[%d]", i);
    init_branch_layer(&net.branch3[i], name, batch_size, 2, ch_in[0], 28, 28);

    sprintf(name, "bn3[%d][0]", i);
    init_bn_layer(&net.bn3[i][0], name, batch_size, ch_in[0], 28, 28);

    sprintf(name, "relu3[%d][0]", i);
    init_act_layer(&net.relu3[i][0], name, batch_size, ch_in[0], 28, 28, RELU_T);

    sprintf(name, "conv3[%d][0]", i);
    init_conv_layer(&net.conv3[i][0], name, batch_size, 1, 1, 0, 0, 1, 1, ch_in[0], 128, 28, 28);

    sprintf(name, "bn3[%d][1]", i);
    init_bn_layer(&net.bn3[i][1], name, batch_size, 128, 28, 28);

    sprintf(name, "relu3[%d][1]", i);
    init_act_layer(&net.relu3[i][1], name, batch_size, 128, 28, 28, RELU_T);

    sprintf(name, "conv3[%d][1]", i);
    init_conv_layer(&net.conv3[i][1], name, batch_size, 3, 3, 1, 1, 1, 1, 128, 32, 28, 28);

    sprintf(name, "concat3[%d]", i);
    init_concat_layer(&net.concat3[i], name, batch_size, 2, ch_in, 28, 28);
  }

  sprintf(name, "trans_bn[1]");
  init_bn_layer(&net.trans_bn[1], name, batch_size, 512, 28, 28); 

  sprintf(name, "trans_relu[1]");
  init_act_layer(&net.trans_relu[1], name, batch_size, 512, 28, 28, RELU_T);

  sprintf(name, "trans_conv[1]");
  init_conv_layer(&net.trans_conv[1], name, batch_size, 1, 1, 0, 0, 1, 1, 512, 256, 28, 28);

  sprintf(name, "pool[2]");
  init_pool_layer(&net.pool[2], name, batch_size, 2, 2, 0, 0, 2, 2, 256, 28, 28, MAX_T);

  for (int i = 0; i < 24; i++) {
    ch_in[0] = 256 + 32 * i;
    ch_in[1] = 32;

    sprintf(name, "branch4[%d]", i);
    init_branch_layer(&net.branch4[i], name, batch_size, 2, ch_in[0], 14, 14);

    sprintf(name, "bn4[%d][0]", i);
    init_bn_layer(&net.bn4[i][0], name, batch_size, ch_in[0], 14, 14);

    sprintf(name, "relu4[%d][0]", i);
    init_act_layer(&net.relu4[i][0], name, batch_size, ch_in[0], 14, 14, RELU_T);

    sprintf(name, "conv4[%d][0]", i);
    init_conv_layer(&net.conv4[i][0], name, batch_size, 1, 1, 0, 0, 1, 1, ch_in[0], 256, 14, 14);

    sprintf(name, "bn4[%d][1]", i);
    init_bn_layer(&net.bn4[i][1], name, batch_size, 256, 14, 14);

    sprintf(name, "relu4[%d][1]", i);
    init_act_layer(&net.relu4[i][1], name, batch_size, 256, 14, 14, RELU_T);

    sprintf(name, "conv4[%d][1]", i);
    init_conv_layer(&net.conv4[i][1], name, batch_size, 3, 3, 1, 1, 1, 1, 256, 32, 14, 14);

    sprintf(name, "concat4[%d]", i);
    init_concat_layer(&net.concat4[i], name, batch_size, 2, ch_in, 14, 14);
  }

  sprintf(name, "trans_bn[2]");
  init_bn_layer(&net.trans_bn[2], name, batch_size, 1024, 14, 14); 

  sprintf(name, "trans_relu[2]");
  init_act_layer(&net.trans_relu[2], name, batch_size, 1024, 14, 14, RELU_T);

  sprintf(name, "trans_conv[2]");
  init_conv_layer(&net.trans_conv[2], name, batch_size, 1, 1, 0, 0, 1, 1, 1024, 512, 14, 14);

  sprintf(name, "pool[3]");
  init_pool_layer(&net.pool[3], name, batch_size, 2, 2, 0, 0, 2, 2, 512, 14, 14, MAX_T);

  for (int i = 0; i < 16; i++) {
    ch_in[0] = 512 + 32 * i;
    ch_in[1] = 32;

    sprintf(name, "branch5[%d]", i);
    init_branch_layer(&net.branch5[i], name, batch_size, 2, ch_in[0], 7, 7);

    sprintf(name, "bn5[%d][0]", i);
    init_bn_layer(&net.bn5[i][0], name, batch_size, ch_in[0], 7, 7);

    sprintf(name, "relu5[%d][0]", i);
    init_act_layer(&net.relu5[i][0], name, batch_size, ch_in[0], 7, 7, RELU_T);

    sprintf(name, "conv5[%d][0]", i);
    init_conv_layer(&net.conv5[i][0], name, batch_size, 1, 1, 0, 0, 1, 1, ch_in[0], 512, 7, 7);

    sprintf(name, "bn5[%d][1]", i);
    init_bn_layer(&net.bn5[i][1], name, batch_size, 512, 7, 7);

    sprintf(name, "relu5[%d][1]", i);
    init_act_layer(&net.relu5[i][1], name, batch_size, 512, 7, 7, RELU_T);

    sprintf(name, "conv5[%d][1]", i);
    init_conv_layer(&net.conv5[i][1], name, batch_size, 3, 3, 1, 1, 1, 1, 512, 32, 7, 7);

    sprintf(name, "concat5[%d]", i);
    init_concat_layer(&net.concat5[i], name, batch_size, 2, ch_in, 7, 7);
  }

  sprintf(name, "trans_bn[3]");
  init_bn_layer(&net.trans_bn[3], name, batch_size, 1024, 7, 7); 

  sprintf(name, "trans_relu[3]");
  init_act_layer(&net.trans_relu[3], name, batch_size, 1024, 7, 7, RELU_T);

  sprintf(name, "pool[4]");
  init_pool_layer(&net.pool[4], name, batch_size, 7, 7, 0, 0, 1, 1, 1024, 7, 7, MAX_T);

  sprintf(name, "fc");
  init_conv_layer(&net.fc, name, batch_size, 1, 1, 0, 0, 1, 1, 1024, 1000, 1, 1);

  sprintf(name, "bias");
  init_bias_layer(&net.bias, name, batch_size, 1000, 1, 1);

  sprintf(name, "softmax");
  init_softmax_layer(&net.softmax, name, batch_size, 1000);

  net.is_initiated = true;
}

#define DENSENET_PARAM(FUNC) \
do {\
  FUNC##_CONV(&net.conv1);\
  FUNC##_BN(&net.bn1);\
  for (int i = 0; i < 3; i++) {\
    FUNC##_BN(&net.trans_bn[i]);\
    FUNC##_CONV(&net.trans_conv[i]);\
  }\
  for (int i = 0; i < 6; i++)\
    for (int j = 0; j < 2; j++) {\
       FUNC##_BN(&net.bn2[i][j]);\
       FUNC##_CONV(&net.conv2[i][j]);\
    }\
  for (int i = 0; i < 12; i++)\
    for (int j = 0; j < 2; j++) {\
       FUNC##_BN(&net.bn3[i][j]);\
       FUNC##_CONV(&net.conv3[i][j]);\
    }\
  for (int i = 0; i < 24; i++)\
    for (int j = 0; j < 2; j++) {\
       FUNC##_BN(&net.bn4[i][j]);\
       FUNC##_CONV(&net.conv4[i][j]);\
    }\
  for (int i = 0; i < 16; i++)\
    for (int j = 0; j < 2; j++) {\
       FUNC##_BN(&net.bn5[i][j]);\
       FUNC##_CONV(&net.conv5[i][j]);\
    }\
  FUNC##_CONV(&net.fc);\
  FUNC##_BIAS(&net.bias);\
} while (0)

size_t densenet_get_param_size()
{
  size_t sum = 0;

  DENSENET_PARAM(SIZE);

  return sum;
}

void densenet_init_param(float *param)
{
  DENSENET_PARAM(INIT);
}

void densenet_get_param(float *param)
{
  DENSENET_PARAM(GET);
}

void densenet_copy_input(float *data_in, int *label_in)
{
  set_input(&net.input, data_in);
  set_label(&net.softmax, label_in);
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

  for (int i = 23; i >= 0; i--) {
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
  CONNECT_FROM_INPUT(net.input, net.conv1);
  CONNECT(net.conv1, net.bn1);
  CONNECT(net.bn1, net.relu1);
  CONNECT(net.relu1, net.pool[0]);

  for (int i = 0; i < 6; i++) {
    if (i == 0) {
      CONNECT(net.pool[0], net.branch2[i]);
    }
    else {
      CONNECT_FROM_CONCAT(net.concat2[i-1], net.branch2[i]);
    }

    CONNECT_FROM_BRANCH_TO_CONCAT(net.branch2[i], net.concat2[i]);
    CONNECT_FROM_BRANCH(net.branch2[i], net.bn2[i][0], 1);

    CONNECT(net.bn2[i][0], net.relu2[i][0]);
    CONNECT(net.relu2[i][0], net.conv2[i][0]);
    CONNECT(net.conv2[i][0], net.bn2[i][1]);
    CONNECT(net.bn2[i][1], net.relu2[i][1]);
    CONNECT(net.relu2[i][1], net.conv2[i][1]);

    CONNECT_TO_CONCAT(net.conv2[i][1], net.concat2[i], 1);
  }

  CONNECT_FROM_CONCAT(net.concat2[5], net.trans_bn[0]);
  CONNECT(net.trans_bn[0], net.trans_relu[0]);
  CONNECT(net.trans_relu[0], net.trans_conv[0]);
  CONNECT(net.trans_conv[0], net.pool[1]);

  for (int i = 0; i < 12; i++) {
    if (i == 0) {
      CONNECT(net.pool[1], net.branch3[i]);
    }
    else {
      CONNECT_FROM_CONCAT(net.concat3[i-1], net.branch3[i]);
    }

    CONNECT_FROM_BRANCH_TO_CONCAT(net.branch3[i], net.concat3[i]);
    CONNECT_FROM_BRANCH(net.branch3[i], net.bn3[i][0], 1);

    CONNECT(net.bn3[i][0], net.relu3[i][0]);
    CONNECT(net.relu3[i][0], net.conv3[i][0]);
    CONNECT(net.conv3[i][0], net.bn3[i][1]);
    CONNECT(net.bn3[i][1], net.relu3[i][1]);
    CONNECT(net.relu3[i][1], net.conv3[i][1]);

    CONNECT_TO_CONCAT(net.conv3[i][1], net.concat3[i], 1);
  }

  CONNECT_FROM_CONCAT(net.concat3[11], net.trans_bn[1]);
  CONNECT(net.trans_bn[1], net.trans_relu[1]);
  CONNECT(net.trans_relu[1], net.trans_conv[1]);
  CONNECT(net.trans_conv[1], net.pool[2]);

  for (int i = 0; i < 24; i++) {
    if (i == 0) {
      CONNECT(net.pool[2], net.branch4[i]);
    }
    else {
      CONNECT_FROM_CONCAT(net.concat4[i-1], net.branch4[i]);
    }

    CONNECT_FROM_BRANCH_TO_CONCAT(net.branch4[i], net.concat4[i]);
    CONNECT_FROM_BRANCH(net.branch4[i], net.bn4[i][0], 1);

    CONNECT(net.bn4[i][0], net.relu4[i][0]);
    CONNECT(net.relu4[i][0], net.conv4[i][0]);
    CONNECT(net.conv4[i][0], net.bn4[i][1]);
    CONNECT(net.bn4[i][1], net.relu4[i][1]);
    CONNECT(net.relu4[i][1], net.conv4[i][1]);

    CONNECT_TO_CONCAT(net.conv4[i][1], net.concat4[i], 1);
  }

  CONNECT_FROM_CONCAT(net.concat4[23], net.trans_bn[2]);
  CONNECT(net.trans_bn[2], net.trans_relu[2]);
  CONNECT(net.trans_relu[2], net.trans_conv[2]);
  CONNECT(net.trans_conv[2], net.pool[3]);

  for (int i = 0; i < 16; i++) {
    if (i == 0) {
      CONNECT(net.pool[3], net.branch5[i]);
    }
    else {
      CONNECT_FROM_CONCAT(net.concat5[i-1], net.branch5[i]);
    }

    CONNECT_FROM_BRANCH_TO_CONCAT(net.branch5[i], net.concat5[i]);
    CONNECT_FROM_BRANCH(net.branch5[i], net.bn5[i][0], 1);

    CONNECT(net.bn5[i][0], net.relu5[i][0]);
    CONNECT(net.relu5[i][0], net.conv5[i][0]);
    CONNECT(net.conv5[i][0], net.bn5[i][1]);
    CONNECT(net.bn5[i][1], net.relu5[i][1]);
    CONNECT(net.relu5[i][1], net.conv5[i][1]);

    CONNECT_TO_CONCAT(net.conv5[i][1], net.concat5[i], 1);
  }

  CONNECT_FROM_CONCAT(net.concat5[15], net.trans_bn[3]);
  CONNECT(net.trans_bn[3], net.trans_relu[3]);
  CONNECT(net.trans_relu[3], net.pool[4]);
  CONNECT(net.pool[4], net.fc);
  CONNECT_WITH_BIAS(net.fc, net.bias, net.softmax);
}

#define DENSENET_LAYER(FUNC) \
do {\
  FUNC##_conv_layer(&net.conv1);\
  FUNC##_bn_layer(&net.bn1);\
  FUNC##_act_layer(&net.relu1);\
  FUNC##_pool_layer(&net.pool[0]);\
  for (int i = 0; i < 6; i++) {\
    FUNC##_branch_layer(&net.branch2[i]);\
    FUNC##_bn_layer(&net.bn2[i][0]);\
    FUNC##_act_layer(&net.relu2[i][0]);\
    FUNC##_conv_layer(&net.conv2[i][0]);\
    FUNC##_bn_layer(&net.bn2[i][1]);\
    FUNC##_act_layer(&net.relu2[i][1]);\
    FUNC##_conv_layer(&net.conv2[i][1]);\
    FUNC##_concat_layer(&net.concat2[i]);\
  }\
  FUNC##_bn_layer(&net.trans_bn[0]);\
  FUNC##_act_layer(&net.trans_relu[0]);\
  FUNC##_conv_layer(&net.trans_conv[0]);\
  FUNC##_pool_layer(&net.pool[1]);\
  for (int i = 0; i < 12; i++) {\
    FUNC##_branch_layer(&net.branch3[i]);\
    FUNC##_bn_layer(&net.bn3[i][0]);\
    FUNC##_act_layer(&net.relu3[i][0]);\
    FUNC##_conv_layer(&net.conv3[i][0]);\
    FUNC##_bn_layer(&net.bn3[i][1]);\
    FUNC##_act_layer(&net.relu3[i][1]);\
    FUNC##_conv_layer(&net.conv3[i][1]);\
    FUNC##_concat_layer(&net.concat3[i]);\
  }\
  FUNC##_bn_layer(&net.trans_bn[1]);\
  FUNC##_act_layer(&net.trans_relu[1]);\
  FUNC##_conv_layer(&net.trans_conv[1]);\
  FUNC##_pool_layer(&net.pool[2]);\
  for (int i = 0; i < 24; i++) {\
    FUNC##_branch_layer(&net.branch4[i]);\
    FUNC##_bn_layer(&net.bn4[i][0]);\
    FUNC##_act_layer(&net.relu4[i][0]);\
    FUNC##_conv_layer(&net.conv4[i][0]);\
    FUNC##_bn_layer(&net.bn4[i][1]);\
    FUNC##_act_layer(&net.relu4[i][1]);\
    FUNC##_conv_layer(&net.conv4[i][1]);\
    FUNC##_concat_layer(&net.concat4[i]);\
  }\
  FUNC##_bn_layer(&net.trans_bn[2]);\
  FUNC##_act_layer(&net.trans_relu[2]);\
  FUNC##_conv_layer(&net.trans_conv[2]);\
  FUNC##_pool_layer(&net.pool[3]);\
  for (int i = 0; i < 16; i++) {\
    FUNC##_branch_layer(&net.branch5[i]);\
    FUNC##_bn_layer(&net.bn5[i][0]);\
    FUNC##_act_layer(&net.relu5[i][0]);\
    FUNC##_conv_layer(&net.conv5[i][0]);\
    FUNC##_bn_layer(&net.bn5[i][1]);\
    FUNC##_act_layer(&net.relu5[i][1]);\
    FUNC##_conv_layer(&net.conv5[i][1]);\
    FUNC##_concat_layer(&net.concat5[i]);\
  }\
  FUNC##_bn_layer(&net.trans_bn[3]);\
  FUNC##_act_layer(&net.trans_relu[3]);\
  FUNC##_pool_layer(&net.pool[4]);\
  FUNC##_conv_layer(&net.fc);\
  FUNC##_bias_layer(&net.bias);\
  FUNC##_softmax_layer(&net.softmax);\
} while (0)

void densenet_clear_time()
{
  DENSENET_LAYER(clear_time);
}

void densenet_print_time()
{
  printf("name, fwd, bwd_data, bwd_weight, update\n");

  DENSENET_LAYER(print_time);
}

void cnn_train(int num_train_image, float *train_data, int *train_label) 
{
  assert(num_train_image % params.batch_size == 0); 

  __init_stream_executer();
  __init_object_manager();

  densenet_init(params.batch_size);
  densenet_connect();

  alloc_buffer_by_type(WORK_SPACE);

  int num_batches = num_train_image / params.batch_size;
  fprintf(stderr, "total iteration : %d\n", num_batches);

  size_t sz = densenet_get_param_size();
  float *param_in = (float *)malloc(sz);
  float *param_out = (float *)malloc(sz);
  float *param_result = (float *)malloc(sz);

  INITIALIZE_RAND(param_in, sz/sizeof(float));
  densenet_init_param(param_in);

  bool is_first = true;
  bool synch_iteration = true;

  enum { FULL_ITERATION, HEAD_ITERATION, COPY_INPUT, NUM_TIMERS };
  struct timespec t_begin[NUM_TIMERS];
  struct timespec t_end[NUM_TIMERS];
  float elapsed_time[NUM_TIMERS] = { 0.0 };

  clock_gettime(CLOCK_MONOTONIC, &t_begin[FULL_ITERATION]);

  for (int e = 0; e < params.epochs; e++) {
    fprintf(stderr, "epoch %d/%d start\n", e+1, params.epochs);

    float *data_in = NULL;
    int *label_in = NULL;

    for (int b = 0; b < num_batches; b++) {
      if (synch_iteration) {
        clock_gettime(CLOCK_MONOTONIC, &t_begin[COPY_INPUT]);
      }

      data_in = train_data + b * params.batch_size * params.width * params.height * params.channel;
      label_in = train_label + b * params.batch_size;

      densenet_copy_input(data_in, label_in);

      if (synch_iteration) {
        clock_gettime(CLOCK_MONOTONIC, &t_end[COPY_INPUT]);
        elapsed_time[COPY_INPUT] += diff_timespec_ms(t_begin[COPY_INPUT], t_end[COPY_INPUT]);
      }

      if (is_first) {
        clock_gettime(CLOCK_MONOTONIC, &t_begin[HEAD_ITERATION]);
      }

      densenet_forward();

#ifdef PRINT_LOSS
      float l = get_loss(&net.softmax, label_in);
      /* MPI */
      if (node_id == 0) {
        printf("loss for %d/%d : %f\n", b, num_batches, l);
      }
#endif

      densenet_backward();

      if (is_first) {
        synch_device();
        clock_gettime(CLOCK_MONOTONIC, &t_end[HEAD_ITERATION]);
        elapsed_time[HEAD_ITERATION] = diff_timespec_ms(t_begin[HEAD_ITERATION], t_end[HEAD_ITERATION]);
#ifdef TIME_LAYER
        densenet_clear_time();
#endif
        is_first = false;
      }
      else if (synch_iteration) {
        synch_device();
      }
    }
  }

  synch_device();
  clock_gettime(CLOCK_MONOTONIC, &t_end[FULL_ITERATION]);
  elapsed_time[FULL_ITERATION] = diff_timespec_ms(t_begin[FULL_ITERATION], t_end[FULL_ITERATION]);

  /* MPI */
  if (node_id == 0) {
    float training_time = elapsed_time[FULL_ITERATION] - elapsed_time[COPY_INPUT];
    float first_training_time = elapsed_time[HEAD_ITERATION];

    fprintf(stderr, "(Excl. 1st iter) %.3f ms, %.3f image / sec\n",
        training_time - first_training_time,
        ((float)(params.batch_size * (params.num_batch_per_epoch * params.epochs - 1)) * 1000 / (training_time - first_training_time)));
    fprintf(stderr, "(Incl. 1st iter) %.3f ms, %.3f image / sec\n",
        training_time, ((float)(params.batch_size * params.num_batch_per_epoch * params.epochs) * 1000 / (training_time)));

#ifdef TIME_LAYER
    densenet_print_time();
#endif
  }

  densenet_get_param(param_out);

  /* MPI */
  if (node_id == 0) {
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

  __finalize_object_manager();
  __finalize_stream_executer();
}

