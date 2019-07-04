#include <stdio.h>
#include <string.h>
#include <stdbool.h>
#include <assert.h>

#include "cnn.h"
#include "params.h"
#include "layer.h"
#include "utils.h"
#include "execute.h"

#define RESNET50
//#define RESNET101
//#define RESNET152

#if defined(RESNET50)
#define B0 3
#define B1 4
#define B2 6
#define B3 4
#elif defined(RESNET101)
#define B0 3
#define B1 4
#define B2 23
#define B3 4
#elif defined(RESNET152)
#define B0 3
#define B1 8
#define B2 36
#define B3 4
#else
#define B0 3
#define B0 4
#define B0 6
#define B0 4
#endif

typedef struct resnet_s {
  input_layer input;

  conv_layer conv1;
  bn_layer conv1_bn;
  act_layer conv1_relu;
  pool_layer pool1, pool2;

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
  elt_layer conv2_add[B0];
  act_layer conv2_relu[B0][3];
  branch_layer branch2[B0];

  conv_layer conv3[B1][3];
  bn_layer conv3_bn[B1][3];
  elt_layer conv3_add[B1];
  act_layer conv3_relu[B1][3];
  branch_layer branch3[B1];

  conv_layer conv4[B2][3];
  bn_layer conv4_bn[B2][3];
  elt_layer conv4_add[B2];
  act_layer conv4_relu[B2][3];
  branch_layer branch4[B2];

  conv_layer conv5[B3][3];
  bn_layer conv5_bn[B3][3];
  elt_layer conv5_add[B3];
  act_layer conv5_relu[B3][3];
  branch_layer branch5[B3];

  conv_layer fc;
  bias_layer bias;

  softmax_layer softmax;

  bool is_initiated;
} resnet;

resnet net = {
  .is_initiated = false
};

void params_modify()
{
}

void resnet_init(int batch_size)
{
  char name[1024];

  srand(params.seed);

  sprintf(name, "input");
  init_input_layer(&net.input, name, batch_size, 3, 224, 224);

  sprintf(name, "conv1");
  init_conv_layer(&net.conv1, name, batch_size, 7, 7, 3, 3, 2, 2, 3, 64, 224, 224);

  sprintf(name, "conv1_bn");
  init_bn_layer(&net.conv1_bn, name, batch_size, 64, 112, 112);

  sprintf(name, "conv1_relu");
  init_act_layer(&net.conv1_relu, name, batch_size, 64, 112, 112, RELU_T);

  sprintf(name, "pool1");
  init_pool_layer(&net.pool1, name, batch_size, 3, 3, 1, 1, 2, 2, 64, 112, 112, MAX_T);

  sprintf(name, "conv2_branch");
  init_conv_layer(&net.conv2_branch, name, batch_size, 1, 1, 0, 0, 1, 1, 64, 256, 56, 56);

  sprintf(name, "conv2_branch_bn");
  init_bn_layer(&net.conv2_branch_bn, name, batch_size, 256, 56, 56);

  for (int i = 0; i < B0; i++) {
    sprintf(name, "branch2[%d]", i);
    if (i == 0) {
      init_branch_layer(&net.branch2[i], name, batch_size, 2, 64, 56, 56);
    }
    else {
      init_branch_layer(&net.branch2[i], name, batch_size, 2, 256, 56, 56);
    }

    sprintf(name, "conv2[%d][0]", i);
    if (i == 0) {
      init_conv_layer(&net.conv2[i][0], name, batch_size, 1, 1, 0, 0, 1, 1, 64, 64, 56, 56);
    }
    else {
      init_conv_layer(&net.conv2[i][0], name, batch_size, 1, 1, 0, 0, 1, 1, 256, 64, 56, 56);
    }

    sprintf(name, "conv2_bn[%d][0]", i);
    init_bn_layer(&net.conv2_bn[i][0], name, batch_size, 64, 56, 56);

    sprintf(name, "conv2_relu[%d][0]", i);
    init_act_layer(&net.conv2_relu[i][0], name, batch_size, 64, 56, 56, RELU_T);

    sprintf(name, "conv2[%d][1]", i);
    init_conv_layer(&net.conv2[i][1], name, batch_size, 3, 3, 1, 1, 1, 1, 64, 64, 56, 56);

    sprintf(name, "conv2_bn[%d][1]", i);
    init_bn_layer(&net.conv2_bn[i][1], name, batch_size, 64, 56, 56);

    sprintf(name, "conv2_relu[%d][1]", i);
    init_act_layer(&net.conv2_relu[i][1], name, batch_size, 64, 56, 56, RELU_T);

    sprintf(name, "conv2[%d][2]", i);
    init_conv_layer(&net.conv2[i][2], name, batch_size, 1, 1, 0, 0, 1, 1, 64, 256, 56, 56);

    sprintf(name, "conv2_bn[%d][1]", i);
    init_bn_layer(&net.conv2_bn[i][2], name, batch_size, 256, 56, 56);

    sprintf(name, "conv2_relu[%d][1]", i);
    init_act_layer(&net.conv2_relu[i][2], name, batch_size, 256, 56, 56, RELU_T);

    sprintf(name, "conv2_add[%d]", i);
    init_elt_layer(&net.conv2_add[i], name, batch_size, 256, 56, 56, ADD_T);
  }

  sprintf(name, "conv3_branch");
  init_conv_layer(&net.conv3_branch, name, batch_size, 1, 1, 0, 0, 2, 2, 256, 512, 56, 56);

  sprintf(name, "conv3_branch_bn");
  init_bn_layer(&net.conv3_branch_bn, name, batch_size, 512, 28, 28);

  for (int i = 0; i < B1; i++) {
    if (i == 0) {
      sprintf(name, "branch3[%d]", i);
      init_branch_layer(&net.branch3[i], name, batch_size, 2, 256, 56, 56);

      sprintf(name, "conv3[%d][0]", i);
      init_conv_layer(&net.conv3[i][0], name, batch_size, 1, 1, 0, 0, 2, 2, 256, 128, 56, 56);
    }
    else {
      sprintf(name, "branch3[%d]", i);
      init_branch_layer(&net.branch3[i], name, batch_size, 2, 512, 28, 28);

      sprintf(name, "conv3[%d][0]", i);
      init_conv_layer(&net.conv3[i][0], name, batch_size, 1, 1, 0, 0, 1, 1, 512, 128, 28, 28);
    }

    sprintf(name, "conv3_bn[%d][0]", i);
    init_bn_layer(&net.conv3_bn[i][0], name, batch_size, 128, 28, 28);

    sprintf(name, "conv3_relu[%d][0]", i);
    init_act_layer(&net.conv3_relu[i][0], name, batch_size, 128, 28, 28, RELU_T);

    sprintf(name, "conv3[%d][1]", i);
    init_conv_layer(&net.conv3[i][1], name, batch_size, 3, 3, 1, 1, 1, 1, 128, 128, 28, 28);

    sprintf(name, "conv3_bn[%d][1]", i);
    init_bn_layer(&net.conv3_bn[i][1], name, batch_size, 128, 28, 28);

    sprintf(name, "conv3_relu[%d][1]", i);
    init_act_layer(&net.conv3_relu[i][1], name, batch_size, 128, 28, 28, RELU_T);

    sprintf(name, "conv3[%d][2]", i);
    init_conv_layer(&net.conv3[i][2], name, batch_size, 1, 1, 0, 0, 1, 1, 128, 512, 28, 28);

    sprintf(name, "conv3_bn[%d][2]", i);
    init_bn_layer(&net.conv3_bn[i][2], name, batch_size, 512, 28, 28);

    sprintf(name, "conv3_relu[%d][2]", i);
    init_act_layer(&net.conv3_relu[i][2], name, batch_size, 512, 28, 28, RELU_T);

    sprintf(name, "conv3_add[%d]", i);
    init_elt_layer(&net.conv3_add[i], name, batch_size, 512, 28, 28, ADD_T);
  }

  sprintf(name, "conv4_branch");
  init_conv_layer(&net.conv4_branch, name, batch_size, 1, 1, 0, 0, 2, 2, 512, 1024, 28, 28);

  sprintf(name, "conv4_branch_bn");
  init_bn_layer(&net.conv4_branch_bn, name, batch_size, 1024, 14, 14);

  for (int i = 0; i < B2; i++) {
    if (i == 0) {
      sprintf(name, "branch4[%d]", i);
      init_branch_layer(&net.branch4[i], name, batch_size, 2, 512, 28, 28);

      sprintf(name, "conv4[%d][0]", i);
      init_conv_layer(&net.conv4[i][0], name, batch_size, 1, 1, 0, 0, 2, 2, 512, 256, 28, 28);
    }
    else {
      sprintf(name, "branch4[%d]", i);
      init_branch_layer(&net.branch4[i], name, batch_size, 2, 1024, 14, 14);

      sprintf(name, "conv4[%d][0]", i);
      init_conv_layer(&net.conv4[i][0], name, batch_size, 1, 1, 0, 0, 1, 1, 1024, 256, 14, 14);
    }

    sprintf(name, "conv4_bn[%d][0]", i);
    init_bn_layer(&net.conv4_bn[i][0], name, batch_size, 256, 14, 14);

    sprintf(name, "conv4_relu[%d][0]", i);
    init_act_layer(&net.conv4_relu[i][0], name, batch_size, 256, 14, 14, RELU_T);

    sprintf(name, "conv4[%d][1]", i);
    init_conv_layer(&net.conv4[i][1], name, batch_size, 3, 3, 1, 1, 1, 1, 256, 256, 14, 14);

    sprintf(name, "conv4_bn[%d][1]", i);
    init_bn_layer(&net.conv4_bn[i][1], name, batch_size, 256, 14, 14);

    sprintf(name, "conv4_relu[%d][1]", i);
    init_act_layer(&net.conv4_relu[i][1], name, batch_size, 256, 14, 14, RELU_T);

    sprintf(name, "conv4[%d][2]", i);
    init_conv_layer(&net.conv4[i][2], name, batch_size, 1, 1, 0, 0, 1, 1, 256, 1024, 14, 14);

    sprintf(name, "conv4_bn[%d][2]", i);
    init_bn_layer(&net.conv4_bn[i][2], name, batch_size, 1024, 14, 14);

    sprintf(name, "conv4_relu[%d][2]", i);
    init_act_layer(&net.conv4_relu[i][2], name, batch_size, 1024, 14, 14, RELU_T);

    sprintf(name, "conv4_add[%d]", i);
    init_elt_layer(&net.conv4_add[i], name, batch_size, 1024, 14, 14, ADD_T);
  }

  sprintf(name, "conv5_branch");
  init_conv_layer(&net.conv5_branch, name, batch_size, 1, 1, 0, 0, 2, 2, 1024, 2048, 14, 14);

  sprintf(name, "conv5_branch_bn");
  init_bn_layer(&net.conv5_branch_bn, name, batch_size, 2048, 7, 7);

  for (int i = 0; i < B3; i++) {
    if (i == 0) {
      sprintf(name, "branch5[%d]", i);
      init_branch_layer(&net.branch5[i], name, batch_size, 2, 1024, 14, 14);

      sprintf(name, "conv5[%d][0]", i);
      init_conv_layer(&net.conv5[i][0], name, batch_size, 1, 1, 0, 0, 2, 2, 1024, 512, 14, 14);
    }
    else {
      sprintf(name, "branch5[%d]", i);
      init_branch_layer(&net.branch5[i], name, batch_size, 2, 2048, 7, 7);

      sprintf(name, "conv5[%d][0]", i);
      init_conv_layer(&net.conv5[i][0], name, batch_size, 1, 1, 0, 0, 1, 1, 2048, 512, 7, 7);
    }

    sprintf(name, "conv5_bn[%d][0]", i);
    init_bn_layer(&net.conv5_bn[i][0], name, batch_size, 512, 7, 7);

    sprintf(name, "conv5_relu[%d][0]", i);
    init_act_layer(&net.conv5_relu[i][0], name, batch_size, 512, 7, 7, RELU_T);

    sprintf(name, "conv5[%d][1]", i);
    init_conv_layer(&net.conv5[i][1], name, batch_size, 3, 3, 1, 1, 1, 1, 512, 512, 7, 7);

    sprintf(name, "conv5_bn[%d][1]", i);
    init_bn_layer(&net.conv5_bn[i][1], name, batch_size, 512, 7, 7);

    sprintf(name, "conv5_relu[%d][1]", i);
    init_act_layer(&net.conv5_relu[i][1], name, batch_size, 512, 7, 7, RELU_T);

    sprintf(name, "conv5[%d][2]", i);
    init_conv_layer(&net.conv5[i][2], name, batch_size, 1, 1, 0, 0, 1, 1, 512, 2048, 7, 7);

    sprintf(name, "conv5_bn[%d][2]", i);
    init_bn_layer(&net.conv5_bn[i][2], name, batch_size, 2048, 7, 7);

    sprintf(name, "conv5_relu[%d][2]", i);
    init_act_layer(&net.conv5_relu[i][2], name, batch_size, 2048, 7, 7, RELU_T);

    sprintf(name, "conv5_add[%d]", i);
    init_elt_layer(&net.conv5_add[i], name, batch_size, 2048, 7, 7, ADD_T);
  }

  sprintf(name, "pool2");
  init_pool_layer(&net.pool2, name, batch_size, 7, 7, 0, 0, 1, 1, 2048, 7, 7, AVERAGE_T);

  sprintf(name, "fc");
  init_conv_layer(&net.fc, name, batch_size, 1, 1, 0, 0, 1, 1, 2048, 1000, 1, 1);

  sprintf(name, "bias");
  init_bias_layer(&net.bias, name, batch_size, 1000, 1, 1);

  sprintf(name, "softmax");
  init_softmax_layer(&net.softmax, name, batch_size, 1000);

  net.is_initiated = true;
}

#define RESNET_PARAM(FUNC) \
do {\
  FUNC##_CONV(&net.conv1);\
  FUNC##_BN(&net.conv1_bn);\
  FUNC##_CONV(&net.conv2_branch);\
  FUNC##_BN(&net.conv2_branch_bn);\
  FUNC##_CONV(&net.conv3_branch);\
  FUNC##_BN(&net.conv3_branch_bn);\
  FUNC##_CONV(&net.conv4_branch);\
  FUNC##_BN(&net.conv4_branch_bn);\
  FUNC##_CONV(&net.conv5_branch);\
  FUNC##_BN(&net.conv5_branch_bn);\
  for (int i = 0; i < B0; i++)\
    for (int j = 0; j < 3; j++) {\
      FUNC##_CONV(&net.conv2[i][j]);\
      FUNC##_BN(&net.conv2_bn[i][j]);\
    }\
  for (int i = 0; i < B1; i++)\
    for (int j = 0; j < 3; j++) {\
      FUNC##_CONV(&net.conv3[i][j]);\
      FUNC##_BN(&net.conv3_bn[i][j]);\
    }\
  for (int i = 0; i < B2; i++)\
    for (int j = 0; j < 3; j++) {\
      FUNC##_CONV(&net.conv4[i][j]);\
      FUNC##_BN(&net.conv4_bn[i][j]);\
    }\
  for (int i = 0; i < B3; i++)\
    for (int j = 0; j < 3; j++) {\
      FUNC##_CONV(&net.conv5[i][j]);\
      FUNC##_BN(&net.conv5_bn[i][j]);\
    }\
  FUNC##_CONV(&net.fc);\
  FUNC##_BIAS(&net.bias);\
} while (0)

size_t resnet_get_param_size()
{
  size_t sum = 0;

  RESNET_PARAM(SIZE);

  return sum;
}

void resnet_load_param(float *param)
{
  RESNET_PARAM(LOAD);
}

void resnet_init_param(float *param)
{
  RESNET_PARAM(INIT);
}

void resnet_get_param(float *param)
{
  RESNET_PARAM(GET);
}

void resnet_copy_input(float *data_in, int *label_in)
{
  set_input(&net.input, data_in);
  set_label(&net.softmax, label_in);
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
      CONNECT_FROM_BRANCH(net.branch2[i], net.conv2_branch, 0);
      CONNECT_FROM_BRANCH(net.branch2[i], net.conv2[i][0], 1);

      CONNECT(net.conv2_branch, net.conv2_branch_bn);

      CONNECT(net.conv2[i][0], net.conv2_bn[i][0]);
      CONNECT(net.conv2_bn[i][0], net.conv2_relu[i][0]);
      CONNECT(net.conv2_relu[i][0], net.conv2[i][1]);
      CONNECT(net.conv2[i][1], net.conv2_bn[i][1]);
      CONNECT(net.conv2_bn[i][1], net.conv2_relu[i][1]);
      CONNECT(net.conv2_relu[i][1], net.conv2[i][2]);
      CONNECT(net.conv2[i][2], net.conv2_bn[i][2]);

      CONNECT_TO_ELT(net.conv2_branch_bn, net.conv2_add[i], 0);
      CONNECT_TO_ELT(net.conv2_bn[i][2], net.conv2_add[i], 1);
      CONNECT(net.conv2_add[i], net.conv2_relu[i][2]);
    }
    else {
      CONNECT(net.conv2_relu[i-1][2], net.branch2[i]);
      CONNECT_FROM_BRANCH_TO_ELT(net.branch2[i], net.conv2_add[i]);
      CONNECT_FROM_BRANCH(net.branch2[i], net.conv2[i][0], 1);

      CONNECT(net.conv2[i][0], net.conv2_bn[i][0]);
      CONNECT(net.conv2_bn[i][0], net.conv2_relu[i][0]);
      CONNECT(net.conv2_relu[i][0], net.conv2[i][1]);
      CONNECT(net.conv2[i][1], net.conv2_bn[i][1]);
      CONNECT(net.conv2_bn[i][1], net.conv2_relu[i][1]);
      CONNECT(net.conv2_relu[i][1], net.conv2[i][2]);
      CONNECT(net.conv2[i][2], net.conv2_bn[i][2]);

      CONNECT_TO_ELT(net.conv2_bn[i][2], net.conv2_add[i], 1);
      CONNECT(net.conv2_add[i], net.conv2_relu[i][2]);
    }
  }

  for (int i = 0; i < B1; i++) {
    if (i == 0) {
      CONNECT(net.conv2_relu[B0-1][2], net.branch3[i]);
      CONNECT_FROM_BRANCH(net.branch3[i], net.conv3_branch, 0);
      CONNECT_FROM_BRANCH(net.branch3[i], net.conv3[i][0], 1);

      CONNECT(net.conv3_branch, net.conv3_branch_bn);

      CONNECT(net.conv3[i][0], net.conv3_bn[i][0]);
      CONNECT(net.conv3_bn[i][0], net.conv3_relu[i][0]);
      CONNECT(net.conv3_relu[i][0], net.conv3[i][1]);
      CONNECT(net.conv3[i][1], net.conv3_bn[i][1]);
      CONNECT(net.conv3_bn[i][1], net.conv3_relu[i][1]);
      CONNECT(net.conv3_relu[i][1], net.conv3[i][2]);
      CONNECT(net.conv3[i][2], net.conv3_bn[i][2]);

      CONNECT_TO_ELT(net.conv3_branch_bn, net.conv3_add[i], 0);
      CONNECT_TO_ELT(net.conv3_bn[i][2], net.conv3_add[i], 1);
      CONNECT(net.conv3_add[i], net.conv3_relu[i][2]);
    }
    else {
      CONNECT(net.conv3_relu[i-1][2], net.branch3[i]);
      CONNECT_FROM_BRANCH_TO_ELT(net.branch3[i], net.conv3_add[i]);
      CONNECT_FROM_BRANCH(net.branch3[i], net.conv3[i][0], 1);

      CONNECT(net.conv3[i][0], net.conv3_bn[i][0]);
      CONNECT(net.conv3_bn[i][0], net.conv3_relu[i][0]);
      CONNECT(net.conv3_relu[i][0], net.conv3[i][1]);
      CONNECT(net.conv3[i][1], net.conv3_bn[i][1]);
      CONNECT(net.conv3_bn[i][1], net.conv3_relu[i][1]);
      CONNECT(net.conv3_relu[i][1], net.conv3[i][2]);
      CONNECT(net.conv3[i][2], net.conv3_bn[i][2]);

      CONNECT_TO_ELT(net.conv3_bn[i][2], net.conv3_add[i], 1);
      CONNECT(net.conv3_add[i], net.conv3_relu[i][2]);
    }
  }

  for (int i = 0; i < B2; i++) {
    if (i == 0) {
      CONNECT(net.conv3_relu[B1-1][2], net.branch4[i]);
      CONNECT_FROM_BRANCH(net.branch4[i], net.conv4_branch, 0);
      CONNECT_FROM_BRANCH(net.branch4[i], net.conv4[i][0], 1);

      CONNECT(net.conv4_branch, net.conv4_branch_bn);

      CONNECT(net.conv4[i][0], net.conv4_bn[i][0]);
      CONNECT(net.conv4_bn[i][0], net.conv4_relu[i][0]);
      CONNECT(net.conv4_relu[i][0], net.conv4[i][1]);
      CONNECT(net.conv4[i][1], net.conv4_bn[i][1]);
      CONNECT(net.conv4_bn[i][1], net.conv4_relu[i][1]);
      CONNECT(net.conv4_relu[i][1], net.conv4[i][2]);
      CONNECT(net.conv4[i][2], net.conv4_bn[i][2]);

      CONNECT_TO_ELT(net.conv4_branch_bn,  net.conv4_add[i], 0);
      CONNECT_TO_ELT(net.conv4_bn[i][2], net.conv4_add[i], 1);
      CONNECT(net.conv4_add[i], net.conv4_relu[i][2]);
    }
    else {
      CONNECT(net.conv4_relu[i-1][2], net.branch4[i]);
      CONNECT_FROM_BRANCH_TO_ELT(net.branch4[i], net.conv4_add[i]);
      CONNECT_FROM_BRANCH(net.branch4[i], net.conv4[i][0], 1);

      CONNECT(net.conv4[i][0], net.conv4_bn[i][0]);
      CONNECT(net.conv4_bn[i][0], net.conv4_relu[i][0]);
      CONNECT(net.conv4_relu[i][0], net.conv4[i][1]);
      CONNECT(net.conv4[i][1], net.conv4_bn[i][1]);
      CONNECT(net.conv4_bn[i][1], net.conv4_relu[i][1]);
      CONNECT(net.conv4_relu[i][1], net.conv4[i][2]);
      CONNECT(net.conv4[i][2], net.conv4_bn[i][2]);

      CONNECT_TO_ELT(net.conv4_bn[i][2], net.conv4_add[i], 1);
      CONNECT(net.conv4_add[i], net.conv4_relu[i][2]);
    }
  }

  for (int i = 0; i < B3; i++) {
    if (i == 0) {
      CONNECT(net.conv4_relu[B2-1][2], net.branch5[i]);
      CONNECT_FROM_BRANCH(net.branch5[i], net.conv5_branch, 0);
      CONNECT_FROM_BRANCH(net.branch5[i], net.conv5[i][0], 1);

      CONNECT(net.conv5_branch, net.conv5_branch_bn);

      CONNECT(net.conv5[i][0], net.conv5_bn[i][0]);
      CONNECT(net.conv5_bn[i][0], net.conv5_relu[i][0]);
      CONNECT(net.conv5_relu[i][0], net.conv5[i][1]);
      CONNECT(net.conv5[i][1], net.conv5_bn[i][1]);
      CONNECT(net.conv5_bn[i][1], net.conv5_relu[i][1]);
      CONNECT(net.conv5_relu[i][1], net.conv5[i][2]);
      CONNECT(net.conv5[i][2], net.conv5_bn[i][2]);

      CONNECT_TO_ELT(net.conv5_branch_bn, net.conv5_add[i], 0);
      CONNECT_TO_ELT(net.conv5_bn[i][2], net.conv5_add[i], 1);
      CONNECT(net.conv5_add[i], net.conv5_relu[i][2]);
    }
    else {
      CONNECT(net.conv5_relu[i-1][2], net.branch5[i]);
      CONNECT_FROM_BRANCH_TO_ELT(net.branch5[i], net.conv5_add[i]);
      CONNECT_FROM_BRANCH(net.branch5[i], net.conv5[i][0], 1);

      CONNECT(net.conv5[i][0], net.conv5_bn[i][0]);
      CONNECT(net.conv5_bn[i][0], net.conv5_relu[i][0]);
      CONNECT(net.conv5_relu[i][0], net.conv5[i][1]);
      CONNECT(net.conv5[i][1], net.conv5_bn[i][1]);
      CONNECT(net.conv5_bn[i][1], net.conv5_relu[i][1]);
      CONNECT(net.conv5_relu[i][1], net.conv5[i][2]);
      CONNECT(net.conv5[i][2], net.conv5_bn[i][2]);

      CONNECT_TO_ELT(net.conv5_bn[i][2], net.conv5_add[i], 1);
      CONNECT(net.conv5_add[i], net.conv5_relu[i][2]);
    }
  }

  CONNECT(net.conv5_relu[B3-1][2], net.pool2);
  CONNECT(net.pool2, net.fc);
  CONNECT_WITH_BIAS(net.fc, net.bias, net.softmax);

  alloc_work_space();
}

#define RESNET_LAYER(FUNC) \
do {\
  FUNC##_conv_layer(&net.conv1);\
  FUNC##_bn_layer(&net.conv1_bn);\
  FUNC##_act_layer(&net.conv1_relu);\
  FUNC##_pool_layer(&net.pool1);\
  for (int i = 0; i < B0; i++) {\
    for (int j = 0; j < 3; j++) {\
      if (j == 0) {\
        if (i == 0) {\
          FUNC##_branch_layer(&net.branch2[i]);\
          FUNC##_conv_layer(&net.conv2_branch);\
          FUNC##_bn_layer(&net.conv2_branch_bn);\
        }\
        else {\
          FUNC##_branch_layer(&net.branch2[i]);\
        }\
      }\
      FUNC##_conv_layer(&net.conv2[i][j]);\
      FUNC##_bn_layer(&net.conv2_bn[i][j]);\
      if (j == 2) {\
        FUNC##_elt_layer(&net.conv2_add[i]);\
      }\
      FUNC##_act_layer(&net.conv2_relu[i][j]);\
    }\
  }\
  for (int i = 0; i < B1; i++) {\
    for (int j = 0; j < 3; j++) {\
      if (j == 0) {\
        if (i == 0) {\
          FUNC##_branch_layer(&net.branch3[i]);\
          FUNC##_conv_layer(&net.conv3_branch);\
          FUNC##_bn_layer(&net.conv3_branch_bn);\
        }\
        else {\
          FUNC##_branch_layer(&net.branch3[i]);\
        }\
      }\
      FUNC##_conv_layer(&net.conv3[i][j]);\
      FUNC##_bn_layer(&net.conv3_bn[i][j]);\
      if (j == 2) {\
        FUNC##_elt_layer(&net.conv3_add[i]);\
      }\
      FUNC##_act_layer(&net.conv3_relu[i][j]);\
    }\
  }\
  for (int i = 0; i < B2; i++) {\
    for (int j = 0; j < 3; j++) {\
      if (j == 0) {\
        if (i == 0) {\
          FUNC##_branch_layer(&net.branch4[i]);\
          FUNC##_conv_layer(&net.conv4_branch);\
          FUNC##_bn_layer(&net.conv4_branch_bn);\
        }\
        else {\
          FUNC##_branch_layer(&net.branch4[i]);\
        }\
      }\
      FUNC##_conv_layer(&net.conv4[i][j]);\
      FUNC##_bn_layer(&net.conv4_bn[i][j]);\
      if (j == 2) {\
        FUNC##_elt_layer(&net.conv4_add[i]);\
      }\
      FUNC##_act_layer(&net.conv4_relu[i][j]);\
    }\
  }\
  for (int i = 0; i < B3; i++) {\
    for (int j = 0; j < 3; j++) {\
      if (j == 0) {\
        if (i == 0) {\
          FUNC##_branch_layer(&net.branch5[i]);\
          FUNC##_conv_layer(&net.conv5_branch);\
          FUNC##_bn_layer(&net.conv5_branch_bn);\
        }\
        else {\
          FUNC##_branch_layer(&net.branch5[i]);\
        }\
      }\
      FUNC##_conv_layer(&net.conv5[i][j]);\
      FUNC##_bn_layer(&net.conv5_bn[i][j]);\
      if (j == 2) {\
        FUNC##_elt_layer(&net.conv5_add[i]);\
      }\
      FUNC##_act_layer(&net.conv5_relu[i][j]);\
    }\
  }\
  FUNC##_pool_layer(&net.pool2);\
  FUNC##_conv_layer(&net.fc);\
  FUNC##_bias_layer(&net.bias);\
  FUNC##_softmax_layer(&net.softmax);\
} while (0)

void resnet_clear_time()
{
  RESNET_LAYER(clear_time);
}

void resnet_print_time()
{
  printf("name, fwd, bwd_data, bwd_weight, update\n");

  RESNET_LAYER(print_time);
}

void cnn_train(int num_train_image, float *train_data, int *train_label) 
{
  assert(num_train_image % params.batch_size == 0); 

  __init_stream_executer();
  __init_object_manager();

  resnet_init(params.batch_size);
  resnet_connect();

  int num_batches = num_train_image / params.batch_size;
  fprintf(stderr, "total iteration : %d\n", num_batches);

  size_t sz = resnet_get_param_size();
  float *param_in = (float *)malloc(sz);
  float *param_out = (float *)malloc(sz);
  float *param_result = (float *)malloc(sz);

  INITIALIZE_RAND(param_in, sz / sizeof(float));
  resnet_init_param(param_in);

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

      data_in = train_data + b * params.batch_size * params.width * params.height * params.channel;
      label_in = train_label + b * params.batch_size;

      resnet_copy_input(data_in, label_in);

      resnet_forward();

#ifdef PRINT_LOSS
      float l = get_loss(&net.softmax, label_in);
      printf("loss for %d/%d : %f\n", b, num_batches, l);
#endif

      resnet_backward();

      if (first) {
        synch_device();
        clock_gettime(CLOCK_MONOTONIC, &ed_f);
#ifdef TIME_LAYER
        resnet_clear_time();
#endif
        first = 0;
      }
    }
  }

  synch_device();
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
