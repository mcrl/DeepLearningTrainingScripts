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

typedef struct inception_s {
  input_layer input;
  
  conv_layer prologue_conv[5];
  bn_layer prologue_bn[5];
  act_layer prologue_relu[5];
  pool_layer prologue_pool[2];
  
  branch_layer A_branch[3];

  conv_layer A_conv1[3];
  bn_layer A_bn1[3];
  act_layer A_relu1[3];

  conv_layer A_conv2[3][2];
  bn_layer A_bn2[3][2];
  act_layer A_relu2[3][2];

  conv_layer A_conv3[3][3];
  bn_layer A_bn3[3][3];
  act_layer A_relu3[3][3];

  pool_layer A_pool4[3];
  conv_layer A_conv4[3];
  bn_layer A_bn4[3];
  act_layer A_relu4[3];

  concat_layer A_concat[3];

  branch_layer B_branch;

  pool_layer B_pool1;
  conv_layer B_conv2;
  bn_layer B_bn2;
  act_layer B_relu2;

  conv_layer B_conv3[3];
  bn_layer B_bn3[3];
  act_layer B_relu3[3];

  concat_layer B_concat;

  branch_layer C_branch[4];

  conv_layer C_conv1[4];
  bn_layer C_bn1[4];
  act_layer C_relu1[4];

  conv_layer C_conv2[4][3];
  bn_layer C_bn2[4][3];
  act_layer C_relu2[4][3];

  conv_layer C_conv3[4][5];
  bn_layer C_bn3[4][5];
  act_layer C_relu3[4][5];

  pool_layer C_pool4[4];
  conv_layer C_conv4[4];
  bn_layer C_bn4[4];
  act_layer C_relu4[4];

  concat_layer C_concat[4];

  branch_layer D_branch;

  pool_layer D_pool1;
  conv_layer D_conv2[2];
  bn_layer D_bn2[2];
  act_layer D_relu2[2];

  conv_layer D_conv3[4];
  bn_layer D_bn3[4];
  act_layer D_relu3[4];

  concat_layer D_concat;

  branch_layer E_branch[2];

  conv_layer E_conv1[2];
  bn_layer E_bn1[2];
  act_layer E_relu1[2];

  conv_layer E_conv2[2][3];
  branch_layer E_branch2[2];
  bn_layer E_bn2[2][3];
  act_layer E_relu2[2][3];

  conv_layer E_conv3[2][4];
  branch_layer E_branch3[2];
  bn_layer E_bn3[2][4];
  act_layer E_relu3[2][4];

  pool_layer E_pool4[2];
  conv_layer E_conv4[2];
  bn_layer E_bn4[2];
  act_layer E_relu4[2];

  concat_layer E_concat[2];

  pool_layer epilogue_pool;
  conv_layer fc;
  bias_layer fc_bias;
  softmax_layer softmax;
} inception;

inception net;

bool is_initiated = false;

void params_modify()
{
  params.height = 299;
  params.width = 299;
}

void inception_init(int batch_size)
{
  chkCUDNN(cudnnCreate(&cudnn));
  chkCUBLAS(cublasCreate(&cublas));
  srand(params.seed);

  init_input_layer(&net.input, cudnn, batch_size, 3, 299, 299);

  init_conv_layer(&net.prologue_conv[0], cudnn, batch_size, 3, 3, 0, 0, 2, 2, 3, 32, 299, 299);
  init_bn_layer(&net.prologue_bn[0], cudnn, batch_size, 32, 149, 149); 
  init_act_layer(&net.prologue_relu[0], cudnn, batch_size, 32, 149, 149);

  init_conv_layer(&net.prologue_conv[1], cudnn, batch_size, 3, 3, 0, 0, 1, 1, 32, 32, 149, 149);
  init_bn_layer(&net.prologue_bn[1], cudnn, batch_size, 32, 147, 147); 
  init_act_layer(&net.prologue_relu[1], cudnn, batch_size, 32, 147, 147);

  init_conv_layer(&net.prologue_conv[2], cudnn, batch_size, 3, 3, 1, 1, 1, 1, 32, 64, 147, 147);
  init_bn_layer(&net.prologue_bn[2], cudnn, batch_size, 64, 147, 147); 
  init_act_layer(&net.prologue_relu[2], cudnn, batch_size, 64, 147, 147);

  init_pool_layer(&net.prologue_pool[0], cudnn, batch_size, 3, 3, 0, 0, 2, 2, 64, 147, 147, max);

  init_conv_layer(&net.prologue_conv[3], cudnn, batch_size, 1, 1, 0, 0, 1, 1, 64, 80, 73, 73);
  init_bn_layer(&net.prologue_bn[3], cudnn, batch_size, 80, 73, 73); 
  init_act_layer(&net.prologue_relu[3], cudnn, batch_size, 80, 73, 73);

  init_conv_layer(&net.prologue_conv[4], cudnn, batch_size, 3, 3, 0, 0, 1, 1, 80, 192, 73, 73);
  init_bn_layer(&net.prologue_bn[4], cudnn, batch_size, 192, 71, 71); 
  init_act_layer(&net.prologue_relu[4], cudnn, batch_size, 192, 71, 71);

  init_pool_layer(&net.prologue_pool[1], cudnn, batch_size, 3, 3, 0, 0, 2, 2, 192, 71, 71, max);
  int prev_channel = 192;

  for (int i = 0; i < 3; i++) {
    int fourth_channel[3] = { 32, 64, 64 };
    int channels[4] = { 64, 64, 96, 32 };

    init_branch_layer(&net.A_branch[i], cudnn, batch_size, 4, prev_channel, 35, 35);

    init_conv_layer(&net.A_conv1[i], cudnn, batch_size, 1, 1, 0, 0, 1, 1, prev_channel, 64, 35, 35);
    init_bn_layer(&net.A_bn1[i], cudnn, batch_size, 64, 35, 35);
    init_act_layer(&net.A_relu1[i], cudnn, batch_size, 64, 35, 35);

    init_conv_layer(&net.A_conv2[i][0], cudnn, batch_size, 1, 1, 0, 0, 1, 1, prev_channel, 48, 35, 35);
    init_bn_layer(&net.A_bn2[i][0], cudnn, batch_size, 48, 35, 35);
    init_act_layer(&net.A_relu2[i][0], cudnn, batch_size, 48, 35, 35);
    init_conv_layer(&net.A_conv2[i][1], cudnn, batch_size, 5, 5, 2, 2, 1, 1, 48, 64, 35, 35);
    init_bn_layer(&net.A_bn2[i][1], cudnn, batch_size, 64, 35, 35);
    init_act_layer(&net.A_relu2[i][1], cudnn, batch_size, 64, 35, 35);

    init_conv_layer(&net.A_conv3[i][0], cudnn, batch_size, 1, 1, 0, 0, 1, 1, prev_channel, 64, 35, 35);
    init_bn_layer(&net.A_bn3[i][0], cudnn, batch_size, 64, 35, 35);
    init_act_layer(&net.A_relu3[i][0], cudnn, batch_size, 64, 35, 35);
    init_conv_layer(&net.A_conv3[i][1], cudnn, batch_size, 3, 3, 1, 1, 1, 1, 64, 96, 35, 35);
    init_bn_layer(&net.A_bn3[i][1], cudnn, batch_size, 96, 35, 35);
    init_act_layer(&net.A_relu3[i][1], cudnn, batch_size, 96, 35, 35);
    init_conv_layer(&net.A_conv3[i][2], cudnn, batch_size, 3, 3, 1, 1, 1, 1, 96, 96, 35, 35);
    init_bn_layer(&net.A_bn3[i][2], cudnn, batch_size, 96, 35, 35);
    init_act_layer(&net.A_relu3[i][2], cudnn, batch_size, 96, 35, 35);

    init_pool_layer(&net.A_pool4[i], cudnn, batch_size, 3, 3, 1, 1, 1, 1, prev_channel, 35, 35, average);
    init_conv_layer(&net.A_conv4[i], cudnn, batch_size, 1, 1, 0, 0, 1, 1, prev_channel, fourth_channel[i], 35, 35);
    init_bn_layer(&net.A_bn4[i], cudnn, batch_size, fourth_channel[i], 35, 35);
    init_act_layer(&net.A_relu4[i], cudnn, batch_size, fourth_channel[i], 35, 35);

    channels[3] = fourth_channel[i];
    init_concat_layer(&net.A_concat[i], cudnn, batch_size, 4, channels, 35, 35);
    prev_channel = 64 + 64 + 96 + fourth_channel[i];
  }

  {
    init_branch_layer(&net.B_branch, cudnn, batch_size, 3, prev_channel, 35, 35);
    init_pool_layer(&net.B_pool1, cudnn, batch_size, 3, 3, 0, 0, 2, 2, prev_channel, 35, 35, max); 

    init_conv_layer(&net.B_conv2, cudnn, batch_size, 3, 3, 0, 0, 2, 2, prev_channel, 384, 35, 35);
    init_bn_layer(&net.B_bn2, cudnn, batch_size, 384, 17, 17);
    init_act_layer(&net.B_relu2, cudnn, batch_size, 384, 17, 17);

    init_conv_layer(&net.B_conv3[0], cudnn, batch_size, 1, 1, 0, 0, 1, 1, prev_channel, 64, 35, 35);
    init_bn_layer(&net.B_bn3[0], cudnn, batch_size, 64, 35, 35);
    init_act_layer(&net.B_relu3[0], cudnn, batch_size, 64, 35, 35);
    init_conv_layer(&net.B_conv3[1], cudnn, batch_size, 3, 3, 1, 1, 1, 1, 64, 96, 35, 35);
    init_bn_layer(&net.B_bn3[1], cudnn, batch_size, 96, 35, 35);
    init_act_layer(&net.B_relu3[1], cudnn, batch_size, 96, 35, 35);
    init_conv_layer(&net.B_conv3[2], cudnn, batch_size, 3, 3, 0, 0, 2, 2, 96, 96, 35, 35);
    init_bn_layer(&net.B_bn3[2], cudnn, batch_size, 96, 17, 17);
    init_act_layer(&net.B_relu3[2], cudnn, batch_size, 96, 17, 17);

    int channels[3] = { 288, 384, 96 };

    init_concat_layer(&net.B_concat, cudnn, batch_size, 3, channels, 17, 17);  
    prev_channel = 768;
  }

  for (int i = 0; i < 4; i++) {
    int variable_channel[4] = { 128, 160, 160, 192 };
    int channels[4] = { 192, 192, 192, 192 };

    init_branch_layer(&net.C_branch[i], cudnn, batch_size, 4, prev_channel, 17, 17);

    init_conv_layer(&net.C_conv1[i], cudnn, batch_size, 1, 1, 0, 0, 1, 1, prev_channel, 192, 17, 17);
    init_bn_layer(&net.C_bn1[i], cudnn, batch_size, 192, 17, 17);
    init_act_layer(&net.C_relu1[i], cudnn, batch_size, 192, 17, 17);

    init_conv_layer(&net.C_conv2[i][0], cudnn, batch_size, 1, 1, 0, 0, 1, 1, prev_channel, variable_channel[i], 17, 17);
    init_bn_layer(&net.C_bn2[i][0], cudnn, batch_size, variable_channel[i], 17, 17);
    init_act_layer(&net.C_relu2[i][0], cudnn, batch_size, variable_channel[i], 17, 17);
    init_conv_layer(&net.C_conv2[i][1], cudnn, batch_size, 1, 7, 0, 3, 1, 1, variable_channel[i], variable_channel[i], 17, 17);
    init_bn_layer(&net.C_bn2[i][1], cudnn, batch_size, variable_channel[i], 17, 17);
    init_act_layer(&net.C_relu2[i][1], cudnn, batch_size, variable_channel[i], 17, 17);
    init_conv_layer(&net.C_conv2[i][2], cudnn, batch_size, 7, 1, 3, 0, 1, 1, variable_channel[i], 192, 17, 17);
    init_bn_layer(&net.C_bn2[i][2], cudnn, batch_size, 192, 17, 17);
    init_act_layer(&net.C_relu2[i][2], cudnn, batch_size, 192, 17, 17);

    init_conv_layer(&net.C_conv3[i][0], cudnn, batch_size, 1, 1, 0, 0, 1, 1, prev_channel, variable_channel[i], 17, 17);
    init_bn_layer(&net.C_bn3[i][0], cudnn, batch_size, variable_channel[i], 17, 17);
    init_act_layer(&net.C_relu3[i][0], cudnn, batch_size, variable_channel[i], 17, 17);
    init_conv_layer(&net.C_conv3[i][1], cudnn, batch_size, 7, 1, 3, 0, 1, 1, variable_channel[i], variable_channel[i], 17, 17);
    init_bn_layer(&net.C_bn3[i][1], cudnn, batch_size, variable_channel[i], 17, 17);
    init_act_layer(&net.C_relu3[i][1], cudnn, batch_size, variable_channel[i], 17, 17);
    init_conv_layer(&net.C_conv3[i][2], cudnn, batch_size, 1, 7, 0, 3, 1, 1, variable_channel[i], variable_channel[i], 17, 17);
    init_bn_layer(&net.C_bn3[i][2], cudnn, batch_size, variable_channel[i], 17, 17);
    init_act_layer(&net.C_relu3[i][2], cudnn, batch_size, variable_channel[i], 17, 17);
    init_conv_layer(&net.C_conv3[i][3], cudnn, batch_size, 7, 1, 3, 0, 1, 1, variable_channel[i], variable_channel[i], 17, 17);
    init_bn_layer(&net.C_bn3[i][3], cudnn, batch_size, variable_channel[i], 17, 17);
    init_act_layer(&net.C_relu3[i][3], cudnn, batch_size, variable_channel[i], 17, 17);
    init_conv_layer(&net.C_conv3[i][4], cudnn, batch_size, 1, 7, 0, 3, 1, 1, variable_channel[i], 192, 17, 17);
    init_bn_layer(&net.C_bn3[i][4], cudnn, batch_size, 192, 17, 17);
    init_act_layer(&net.C_relu3[i][4], cudnn, batch_size, 192, 17, 17);

    init_pool_layer(&net.C_pool4[i], cudnn, batch_size, 3, 3, 1, 1, 1, 1, prev_channel, 17, 17, average);
    init_conv_layer(&net.C_conv4[i], cudnn, batch_size, 1, 1, 0, 0, 1, 1, prev_channel, 192, 17, 17);
    init_bn_layer(&net.C_bn4[i], cudnn, batch_size, 192, 17, 17);
    init_act_layer(&net.C_relu4[i], cudnn, batch_size, 192, 17, 17);

    init_concat_layer(&net.C_concat[i], cudnn, batch_size, 4, channels, 17, 17);
    prev_channel = 768;
  }
   
  {
    init_branch_layer(&net.D_branch, cudnn, batch_size, 3, prev_channel, 17, 17);
    init_pool_layer(&net.D_pool1, cudnn, batch_size, 3, 3, 0, 0, 2, 2, prev_channel, 17, 17, max); 

    init_conv_layer(&net.D_conv2[0], cudnn, batch_size, 1, 1, 0, 0, 1, 1, prev_channel, 192, 17, 17);
    init_bn_layer(&net.D_bn2[0], cudnn, batch_size, 192, 17, 17);
    init_act_layer(&net.D_relu2[0], cudnn, batch_size, 192, 17, 17);
    init_conv_layer(&net.D_conv2[1], cudnn, batch_size, 3, 3, 0, 0, 2, 2, 192, 320, 17, 17);
    init_bn_layer(&net.D_bn2[1], cudnn, batch_size, 320, 8, 8);
    init_act_layer(&net.D_relu2[1], cudnn, batch_size, 320, 8, 8);

    init_conv_layer(&net.D_conv3[0], cudnn, batch_size, 1, 1, 0, 0, 1, 1, prev_channel, 192, 17, 17);
    init_bn_layer(&net.D_bn3[0], cudnn, batch_size, 192, 17, 17);
    init_act_layer(&net.D_relu3[0], cudnn, batch_size, 192, 17, 17);
    init_conv_layer(&net.D_conv3[1], cudnn, batch_size, 1, 7, 0, 3, 1, 1, 192, 192, 17, 17);
    init_bn_layer(&net.D_bn3[1], cudnn, batch_size, 192, 17, 17);
    init_act_layer(&net.D_relu3[1], cudnn, batch_size, 192, 17, 17);
    init_conv_layer(&net.D_conv3[2], cudnn, batch_size, 7, 1, 3, 0, 1, 1, 192, 192, 17, 17);
    init_bn_layer(&net.D_bn3[2], cudnn, batch_size, 192, 17, 17);
    init_act_layer(&net.D_relu3[2], cudnn, batch_size, 192, 17, 17);
    init_conv_layer(&net.D_conv3[3], cudnn, batch_size, 3, 3, 0, 0, 2, 2, 192, 192, 17, 17);
    init_bn_layer(&net.D_bn3[3], cudnn, batch_size, 192, 8, 8);
    init_act_layer(&net.D_relu3[3], cudnn, batch_size, 192, 8, 8);

    int channels[3] = { 768, 320, 192 }; 

    init_concat_layer(&net.D_concat, cudnn, batch_size, 3, channels, 8, 8);  
    prev_channel = 1280;
  }

  for (int i = 0; i < 2; i++) {
    int channels[6] = { 320, 384, 384, 384, 384, 192 };

    init_branch_layer(&net.E_branch[i], cudnn, batch_size, 4, prev_channel, 8, 8);

    init_conv_layer(&net.E_conv1[i], cudnn, batch_size, 1, 1, 0, 0, 1, 1, prev_channel, 320, 8, 8);
    init_bn_layer(&net.E_bn1[i], cudnn, batch_size, 320, 8, 8);
    init_act_layer(&net.E_relu1[i], cudnn, batch_size, 320, 8, 8);

    init_conv_layer(&net.E_conv2[i][0], cudnn, batch_size, 1, 1, 0, 0, 1, 1, prev_channel, 384, 8, 8);
    init_bn_layer(&net.E_bn2[i][0], cudnn, batch_size, 384, 8, 8);
    init_act_layer(&net.E_relu2[i][0], cudnn, batch_size, 384, 8, 8);
    init_branch_layer(&net.E_branch2[i], cudnn, batch_size, 2, 384, 8, 8);
    init_conv_layer(&net.E_conv2[i][1], cudnn, batch_size, 1, 3, 0, 1, 1, 1, 384, 384, 8, 8);
    init_bn_layer(&net.E_bn2[i][1], cudnn, batch_size, 384, 8, 8);
    init_act_layer(&net.E_relu2[i][1], cudnn, batch_size, 384, 8, 8);
    init_conv_layer(&net.E_conv2[i][2], cudnn, batch_size, 3, 1, 1, 0, 1, 1, 384, 384, 8, 8);
    init_bn_layer(&net.E_bn2[i][2], cudnn, batch_size, 384, 8, 8);
    init_act_layer(&net.E_relu2[i][2], cudnn, batch_size, 384, 8, 8);

    init_conv_layer(&net.E_conv3[i][0], cudnn, batch_size, 1, 1, 0, 0, 1, 1, prev_channel, 448, 8, 8);
    init_bn_layer(&net.E_bn3[i][0], cudnn, batch_size, 448, 8, 8);
    init_act_layer(&net.E_relu3[i][0], cudnn, batch_size, 448, 8, 8);
    init_conv_layer(&net.E_conv3[i][1], cudnn, batch_size, 3, 3, 1, 1, 1, 1, 448, 384, 8, 8);
    init_bn_layer(&net.E_bn3[i][1], cudnn, batch_size, 384, 8, 8);
    init_act_layer(&net.E_relu3[i][1], cudnn, batch_size, 384, 8, 8);
    init_branch_layer(&net.E_branch3[i], cudnn, batch_size, 2, 384, 8, 8);
    init_conv_layer(&net.E_conv3[i][2], cudnn, batch_size, 1, 3, 0, 1, 1, 1, 384, 384, 8, 8);
    init_bn_layer(&net.E_bn3[i][2], cudnn, batch_size, 384, 8, 8);
    init_act_layer(&net.E_relu3[i][2], cudnn, batch_size, 384, 8, 8);
    init_conv_layer(&net.E_conv3[i][3], cudnn, batch_size, 3, 1, 1, 0, 1, 1, 384, 384, 8, 8);
    init_bn_layer(&net.E_bn3[i][3], cudnn, batch_size, 384, 8, 8);
    init_act_layer(&net.E_relu3[i][3], cudnn, batch_size, 384, 8, 8);

    init_pool_layer(&net.E_pool4[i], cudnn, batch_size, 3, 3, 1, 1, 1, 1, prev_channel, 8, 8, average);
    init_conv_layer(&net.E_conv4[i], cudnn, batch_size, 1, 1, 0, 0, 1, 1, prev_channel, 192, 8, 8);
    init_bn_layer(&net.E_bn4[i], cudnn, batch_size, 192, 8, 8);
    init_act_layer(&net.E_relu4[i], cudnn, batch_size, 192, 8, 8);

    init_concat_layer(&net.E_concat[i], cudnn, batch_size, 6, channels, 8, 8);
    prev_channel = 2048;
  }

  init_pool_layer(&net.epilogue_pool, cudnn, batch_size, 8, 8, 0, 0, 1, 1, 2048, 8, 8, average);
  init_conv_layer(&net.fc, cudnn, batch_size, 1, 1, 0, 0, 1, 1, 2048, 1000, 1, 1);
  init_bias_layer(&net.fc_bias, cudnn, batch_size, 1000, 1, 1);
  init_softmax_layer(&net.softmax, cudnn, batch_size, 1000);

  init_conv_workspace();

  is_initiated = true;
}

size_t inception_get_param_size()
{
  size_t sum = 0;

  for (int i = 0; i < 5; i++) {
    sum += PSIZE_CONV(net.prologue_conv[i]);
    sum += PSIZE_BN(net.prologue_bn[i]);
  }

  for (int i = 0; i < 3; i++) {
    sum += PSIZE_CONV(net.A_conv1[i]);
    sum += PSIZE_BN(net.A_bn1[i]);

    for (int j = 0; j < 2; j++) {
      sum += PSIZE_CONV(net.A_conv2[i][j]);
      sum += PSIZE_BN(net.A_bn2[i][j]);
    }

    for (int j = 0; j < 3; j++) {
      sum += PSIZE_CONV(net.A_conv3[i][j]);
      sum += PSIZE_BN(net.A_bn3[i][j]);
    }

    sum += PSIZE_CONV(net.A_conv4[i]);
    sum += PSIZE_BN(net.A_bn4[i]);
  }

  {
    sum += PSIZE_CONV(net.B_conv2);
    sum += PSIZE_BN(net.B_bn2);

    for (int j = 0; j < 3; j++) {
      sum += PSIZE_CONV(net.B_conv3[j]);
      sum += PSIZE_BN(net.B_bn3[j]);
    }
  }

  for (int i = 0; i < 4; i++) {
    sum += PSIZE_CONV(net.C_conv1[i]);
    sum += PSIZE_BN(net.C_bn1[i]);

    for (int j = 0; j < 3; j++) {
      sum += PSIZE_CONV(net.C_conv2[i][j]);
      sum += PSIZE_BN(net.C_bn2[i][j]);
    }

    for (int j = 0; j < 5; j++) {
      sum += PSIZE_CONV(net.C_conv3[i][j]);
      sum += PSIZE_BN(net.C_bn3[i][j]);
    }

    sum += PSIZE_CONV(net.C_conv4[i]);
    sum += PSIZE_BN(net.C_bn4[i]);
  }

  {
    for (int j = 0; j < 2; j++) {
      sum += PSIZE_CONV(net.D_conv2[j]);
      sum += PSIZE_BN(net.D_bn2[j]);
    }

    for (int j = 0; j < 4; j++) {
      sum += PSIZE_CONV(net.D_conv3[j]);
      sum += PSIZE_BN(net.D_bn3[j]);
    }
  }

  for (int i = 0; i < 2; i++) {
    sum += PSIZE_CONV(net.E_conv1[i]);
    sum += PSIZE_BN(net.E_bn1[i]);

    for (int j = 0; j < 3; j++) {
      sum += PSIZE_CONV(net.E_conv2[i][j]);
      sum += PSIZE_BN(net.E_bn2[i][j]);
    }

    for (int j = 0; j < 4; j++) {
      sum += PSIZE_CONV(net.E_conv3[i][j]);
      sum += PSIZE_BN(net.E_bn3[i][j]);
    }

    sum += PSIZE_CONV(net.E_conv4[i]);
    sum += PSIZE_BN(net.E_bn4[i]);
  }

  sum += PSIZE_CONV(net.fc);
  sum += PSIZE_BIAS(net.fc_bias);

  return sum;
}

void inception_set_param(float *param)
{
  for (int i = 0; i < 5; i++) {
    INIT_CONV(net.prologue_conv[i]);
    INIT_BN(net.prologue_bn[i]);
  }

  for (int i = 0; i < 3; i++) {
    INIT_CONV(net.A_conv1[i]);
    INIT_BN(net.A_bn1[i]);

    for (int j = 0; j < 2; j++) {
      INIT_CONV(net.A_conv2[i][j]);
      INIT_BN(net.A_bn2[i][j]);
    }

    for (int j = 0; j < 3; j++) {
      INIT_CONV(net.A_conv3[i][j]);
      INIT_BN(net.A_bn3[i][j]);
    }

    INIT_CONV(net.A_conv4[i]);
    INIT_BN(net.A_bn4[i]);
  }

  {
    INIT_CONV(net.B_conv2);
    INIT_BN(net.B_bn2);

    for (int j = 0; j < 3; j++) {
      INIT_CONV(net.B_conv3[j]);
      INIT_BN(net.B_bn3[j]);
    }
  }

  for (int i = 0; i < 4; i++) {
    INIT_CONV(net.C_conv1[i]);
    INIT_BN(net.C_bn1[i]);

    for (int j = 0; j < 3; j++) {
      INIT_CONV(net.C_conv2[i][j]);
      INIT_BN(net.C_bn2[i][j]);
    }

    for (int j = 0; j < 5; j++) {
      INIT_CONV(net.C_conv3[i][j]);
      INIT_BN(net.C_bn3[i][j]);
    }

    INIT_CONV(net.C_conv4[i]);
    INIT_BN(net.C_bn4[i]);
  }

  {
    for (int j = 0; j < 2; j++) {
      INIT_CONV(net.D_conv2[j]);
      INIT_BN(net.D_bn2[j]);
    }

    for (int j = 0; j < 4; j++) {
      INIT_CONV(net.D_conv3[j]);
      INIT_BN(net.D_bn3[j]);
    }
  }

  for (int i = 0; i < 2; i++) {
    INIT_CONV(net.E_conv1[i]);
    INIT_BN(net.E_bn1[i]);

    for (int j = 0; j < 3; j++) {
      INIT_CONV(net.E_conv2[i][j]);
      INIT_BN(net.E_bn2[i][j]);
    }

    for (int j = 0; j < 4; j++) {
      INIT_CONV(net.E_conv3[i][j]);
      INIT_BN(net.E_bn3[i][j]);
    }

    INIT_CONV(net.E_conv4[i]);
    INIT_BN(net.E_bn4[i]);
  }

  INIT_CONV(net.fc);
  INIT_BIAS(net.fc_bias);
}

void inception_get_param(float *param)
{
  for (int i = 0; i < 5; i++) {
    GET_CONV(net.prologue_conv[i]);
    GET_BN(net.prologue_bn[i]);
  }

  for (int i = 0; i < 3; i++) {
    GET_CONV(net.A_conv1[i]);
    GET_BN(net.A_bn1[i]);

    for (int j = 0; j < 2; j++) {
      GET_CONV(net.A_conv2[i][j]);
      GET_BN(net.A_bn2[i][j]);
    }

    for (int j = 0; j < 3; j++) {
      GET_CONV(net.A_conv3[i][j]);
      GET_BN(net.A_bn3[i][j]);
    }

    GET_CONV(net.A_conv4[i]);
    GET_BN(net.A_bn4[i]);
  }

  {
    GET_CONV(net.B_conv2);
    GET_BN(net.B_bn2);

    for (int j = 0; j < 3; j++) {
      GET_CONV(net.B_conv3[j]);
      GET_BN(net.B_bn3[j]);
    }
  }

  for (int i = 0; i < 4; i++) {
    GET_CONV(net.C_conv1[i]);
    GET_BN(net.C_bn1[i]);

    for (int j = 0; j < 3; j++) {
      GET_CONV(net.C_conv2[i][j]);
      GET_BN(net.C_bn2[i][j]);
    }

    for (int j = 0; j < 5; j++) {
      GET_CONV(net.C_conv3[i][j]);
      GET_BN(net.C_bn3[i][j]);
    }

    GET_CONV(net.C_conv4[i]);
    GET_BN(net.C_bn4[i]);
  }

  {
    for (int j = 0; j < 2; j++) {
      GET_CONV(net.D_conv2[j]);
      GET_BN(net.D_bn2[j]);
    }

    for (int j = 0; j < 4; j++) {
      GET_CONV(net.D_conv3[j]);
      GET_BN(net.D_bn3[j]);
    }
  }

  for (int i = 0; i < 2; i++) {
    GET_CONV(net.E_conv1[i]);
    GET_BN(net.E_bn1[i]);

    for (int j = 0; j < 3; j++) {
      GET_CONV(net.E_conv2[i][j]);
      GET_BN(net.E_bn2[i][j]);
    }

    for (int j = 0; j < 4; j++) {
      GET_CONV(net.E_conv3[i][j]);
      GET_BN(net.E_bn3[i][j]);
    }

    GET_CONV(net.E_conv4[i]);
    GET_BN(net.E_bn4[i]);
  }

  GET_CONV(net.fc);
  GET_BIAS(net.fc_bias);
}

void inception_copy_input(int batch_size, float *data_in, int *label_in)
{
  size_t input_size = sizeof(float) * batch_size * params.width * params.height * params.channel;

  chkCUDA(cudaMemcpy(net.input.output, data_in, input_size, cudaMemcpyHostToDevice)); 
  chkCUDA(cudaMemcpy(net.softmax.label_in, label_in, batch_size * sizeof(int), cudaMemcpyHostToDevice)); 
  cuda_set_label(batch_size, 1000, net.softmax.label_in, net.softmax.label);
}

void inception_forward()
{
  for (int i = 0; i < 3; i++) {
    train_fwd_conv_layer(&net.prologue_conv[i]);
    train_fwd_bn_layer(&net.prologue_bn[i]);
    train_fwd_act_layer(&net.prologue_relu[i]);
  }

  train_fwd_pool_layer(&net.prologue_pool[0]);

  for (int i = 3; i < 5; i++) {
    train_fwd_conv_layer(&net.prologue_conv[i]);
    train_fwd_bn_layer(&net.prologue_bn[i]);
    train_fwd_act_layer(&net.prologue_relu[i]);
  }

  train_fwd_pool_layer(&net.prologue_pool[1]);

  for (int i = 0; i < 3; i++) {
    train_fwd_branch_layer(&net.A_branch[i]);
    train_fwd_conv_layer(&net.A_conv1[i]);
    train_fwd_bn_layer(&net.A_bn1[i]);
    train_fwd_act_layer(&net.A_relu1[i]);

    for (int j = 0; j < 2; j++) {
      train_fwd_conv_layer(&net.A_conv2[i][j]);
      train_fwd_bn_layer(&net.A_bn2[i][j]);
      train_fwd_act_layer(&net.A_relu2[i][j]);
    }

    for (int j = 0; j < 3; j++) {
      train_fwd_conv_layer(&net.A_conv3[i][j]);
      train_fwd_bn_layer(&net.A_bn3[i][j]);
      train_fwd_act_layer(&net.A_relu3[i][j]);
    }

    train_fwd_pool_layer(&net.A_pool4[i]);
    train_fwd_conv_layer(&net.A_conv4[i]);
    train_fwd_bn_layer(&net.A_bn4[i]);
    train_fwd_act_layer(&net.A_relu4[i]);

    train_fwd_concat_layer(&net.A_concat[i]);
  }

  {
    train_fwd_branch_layer(&net.B_branch);

    train_fwd_pool_layer(&net.B_pool1);
    train_fwd_conv_layer(&net.B_conv2);
    train_fwd_bn_layer(&net.B_bn2);
    train_fwd_act_layer(&net.B_relu2);

    for (int j = 0; j < 3; j++) {
      train_fwd_conv_layer(&net.B_conv3[j]);
      train_fwd_bn_layer(&net.B_bn3[j]);
      train_fwd_act_layer(&net.B_relu3[j]);
    }

    train_fwd_concat_layer(&net.B_concat);
  }

  for (int i = 0; i < 4; i++) {
    train_fwd_branch_layer(&net.C_branch[i]);
    train_fwd_conv_layer(&net.C_conv1[i]);
    train_fwd_bn_layer(&net.C_bn1[i]);
    train_fwd_act_layer(&net.C_relu1[i]);

    for (int j = 0; j < 3; j++) {
      train_fwd_conv_layer(&net.C_conv2[i][j]);
      train_fwd_bn_layer(&net.C_bn2[i][j]);
      train_fwd_act_layer(&net.C_relu2[i][j]);
    }

    for (int j = 0; j < 5; j++) {
      train_fwd_conv_layer(&net.C_conv3[i][j]);
      train_fwd_bn_layer(&net.C_bn3[i][j]);
      train_fwd_act_layer(&net.C_relu3[i][j]);
    }

    train_fwd_pool_layer(&net.C_pool4[i]);
    train_fwd_conv_layer(&net.C_conv4[i]);
    train_fwd_bn_layer(&net.C_bn4[i]);
    train_fwd_act_layer(&net.C_relu4[i]);

    train_fwd_concat_layer(&net.C_concat[i]);
  }

  {
    train_fwd_branch_layer(&net.D_branch);
    train_fwd_pool_layer(&net.D_pool1);

    for (int j = 0; j < 2; j++) {
      train_fwd_conv_layer(&net.D_conv2[j]);
      train_fwd_bn_layer(&net.D_bn2[j]);
      train_fwd_act_layer(&net.D_relu2[j]);
    }

    for (int j = 0; j < 4; j++) {
      train_fwd_conv_layer(&net.D_conv3[j]);
      train_fwd_bn_layer(&net.D_bn3[j]);
      train_fwd_act_layer(&net.D_relu3[j]);
    }

    train_fwd_concat_layer(&net.D_concat);
  }  
  
  for (int i = 0; i < 2; i++) {
    train_fwd_branch_layer(&net.E_branch[i]);
    train_fwd_conv_layer(&net.E_conv1[i]);
    train_fwd_bn_layer(&net.E_bn1[i]);
    train_fwd_act_layer(&net.E_relu1[i]);

    for (int j = 0; j < 3; j++) {
      train_fwd_conv_layer(&net.E_conv2[i][j]);
      train_fwd_bn_layer(&net.E_bn2[i][j]);
      train_fwd_act_layer(&net.E_relu2[i][j]);

      if (j == 0) {
        train_fwd_branch_layer(&net.E_branch2[i]);
      }
    }

    for (int j = 0; j < 4; j++) {
      train_fwd_conv_layer(&net.E_conv3[i][j]);
      train_fwd_bn_layer(&net.E_bn3[i][j]);
      train_fwd_act_layer(&net.E_relu3[i][j]);

      if (j == 1) {
        train_fwd_branch_layer(&net.E_branch3[i]);
      }
    }

    train_fwd_pool_layer(&net.E_pool4[i]);
    train_fwd_conv_layer(&net.E_conv4[i]);
    train_fwd_bn_layer(&net.E_bn4[i]);
    train_fwd_act_layer(&net.E_relu4[i]);

    train_fwd_concat_layer(&net.E_concat[i]);
  }

  train_fwd_pool_layer(&net.epilogue_pool);
  train_fwd_conv_layer(&net.fc);
  train_fwd_bias_layer(&net.fc_bias);
  train_fwd_softmax_layer(&net.softmax);
}

void inception_backward()
{
  train_bwd_softmax_layer(&net.softmax);
  train_bwd_bias_layer(&net.fc_bias);
  train_bwd_conv_layer(&net.fc);
  train_bwd_pool_layer(&net.epilogue_pool);

  for (int i = 1; i >= 0; i--) {
    train_bwd_concat_layer(&net.E_concat[i]);

    train_bwd_act_layer(&net.E_relu4[i]);
    train_bwd_bn_layer(&net.E_bn4[i]);
    train_bwd_conv_layer(&net.E_conv4[i]);
    train_bwd_pool_layer(&net.E_pool4[i]);

    for (int j = 3; j >= 0; j--) {
      if (j == 1) {
        train_bwd_branch_layer(&net.E_branch3[i]);
      }

      train_bwd_act_layer(&net.E_relu3[i][j]);
      train_bwd_bn_layer(&net.E_bn3[i][j]);
      train_bwd_conv_layer(&net.E_conv3[i][j]);
    }

    for (int j = 2; j >= 0; j--) {
      if (j == 0) {
        train_bwd_branch_layer(&net.E_branch2[i]);
      }

      train_bwd_act_layer(&net.E_relu2[i][j]);
      train_bwd_bn_layer(&net.E_bn2[i][j]);
      train_bwd_conv_layer(&net.E_conv2[i][j]);
    }

    train_bwd_act_layer(&net.E_relu1[i]);
    train_bwd_bn_layer(&net.E_bn1[i]);
    train_bwd_conv_layer(&net.E_conv1[i]);
    train_bwd_branch_layer(&net.E_branch[i]);
  }

  {
    train_bwd_concat_layer(&net.D_concat);

    for (int j = 3; j>=0; j--) {
      train_bwd_act_layer(&net.D_relu3[j]);
      train_bwd_bn_layer(&net.D_bn3[j]);
      train_bwd_conv_layer(&net.D_conv3[j]);
    }

    for (int j = 1; j >= 0; j--) {
      train_bwd_act_layer(&net.D_relu2[j]);
      train_bwd_bn_layer(&net.D_bn2[j]);
      train_bwd_conv_layer(&net.D_conv2[j]);
    }

    train_bwd_pool_layer(&net.D_pool1);
    train_bwd_branch_layer(&net.D_branch);
  }  

  for (int i = 3; i >= 0; i--) {
    train_bwd_concat_layer(&net.C_concat[i]);

    train_bwd_act_layer(&net.C_relu4[i]);
    train_bwd_bn_layer(&net.C_bn4[i]);
    train_bwd_conv_layer(&net.C_conv4[i]);
    train_bwd_pool_layer(&net.C_pool4[i]);

    for (int j = 4; j >= 0; j--) {
      train_bwd_act_layer(&net.C_relu3[i][j]);
      train_bwd_bn_layer(&net.C_bn3[i][j]);
      train_bwd_conv_layer(&net.C_conv3[i][j]);
    }

    for (int j = 2; j >= 0; j--) {
      train_bwd_act_layer(&net.C_relu2[i][j]);
      train_bwd_bn_layer(&net.C_bn2[i][j]);
      train_bwd_conv_layer(&net.C_conv2[i][j]);
    }

    train_bwd_act_layer(&net.C_relu1[i]);
    train_bwd_bn_layer(&net.C_bn1[i]);
    train_bwd_conv_layer(&net.C_conv1[i]);
    train_bwd_branch_layer(&net.C_branch[i]);
  }

  {
    train_bwd_concat_layer(&net.B_concat);

    for (int j = 2; j >=0; j--) {
      train_bwd_act_layer(&net.B_relu3[j]);
      train_bwd_bn_layer(&net.B_bn3[j]);
      train_bwd_conv_layer(&net.B_conv3[j]);
    }

    train_bwd_act_layer(&net.B_relu2);
    train_bwd_bn_layer(&net.B_bn2);
    train_bwd_conv_layer(&net.B_conv2);

    train_bwd_pool_layer(&net.B_pool1);
    train_bwd_branch_layer(&net.B_branch);
  }

  for (int i = 2; i >= 0; i--) {
    train_bwd_concat_layer(&net.A_concat[i]);

    train_bwd_act_layer(&net.A_relu4[i]);
    train_bwd_bn_layer(&net.A_bn4[i]);
    train_bwd_conv_layer(&net.A_conv4[i]);
    train_bwd_pool_layer(&net.A_pool4[i]);

    for (int j = 2; j >= 0; j--) {
      train_bwd_act_layer(&net.A_relu3[i][j]);
      train_bwd_bn_layer(&net.A_bn3[i][j]);
      train_bwd_conv_layer(&net.A_conv3[i][j]);
    }

    for (int j = 1; j >= 0; j--) {
      train_bwd_act_layer(&net.A_relu2[i][j]);
      train_bwd_bn_layer(&net.A_bn2[i][j]);
      train_bwd_conv_layer(&net.A_conv2[i][j]);
    }

    train_bwd_act_layer(&net.A_relu1[i]);
    train_bwd_bn_layer(&net.A_bn1[i]);
    train_bwd_conv_layer(&net.A_conv1[i]);
    train_bwd_branch_layer(&net.A_branch[i]);
  }

  train_bwd_pool_layer(&net.prologue_pool[1]);

  for (int i = 4; i >= 3; i--) {
    train_bwd_act_layer(&net.prologue_relu[i]);
    train_bwd_bn_layer(&net.prologue_bn[i]);
    train_bwd_conv_layer(&net.prologue_conv[i]);
  }

  train_bwd_pool_layer(&net.prologue_pool[0]);

  for (int i = 2; i >= 0; i--) {
    train_bwd_act_layer(&net.prologue_relu[i]);
    train_bwd_bn_layer(&net.prologue_bn[i]);
    train_bwd_conv_layer(&net.prologue_conv[i]);
  }
}

void inception_connect()
{
  CONNECT(net.input, net.prologue_conv[0]);

  for (int i = 0; i < 3; i++) {
    CONNECT(net.prologue_conv[i], net.prologue_bn[i]);
    CONNECT(net.prologue_bn[i], net.prologue_relu[i]);

    if (i != 2) {
      CONNECT(net.prologue_relu[i], net.prologue_conv[i+1]);
    }
    else {
      CONNECT(net.prologue_relu[i], net.prologue_pool[0]);
    }
  }

  CONNECT(net.prologue_pool[0], net.prologue_conv[3]);

  for (int i = 3; i < 5; i++) {
    CONNECT(net.prologue_conv[i], net.prologue_bn[i]);
    CONNECT(net.prologue_bn[i], net.prologue_relu[i]);

    if (i != 4) {
      CONNECT(net.prologue_relu[i], net.prologue_conv[i+1]);
    }
    else {
      CONNECT(net.prologue_relu[i], net.prologue_pool[1]);
    }
  }

  CONNECT(net.prologue_pool[1], net.A_branch[0]);

  for (int i = 0; i < 3; i++) {
    CONNECT_BRANCH_I(net.A_branch[i], net.A_conv1[i], 0);
    CONNECT_BRANCH_I(net.A_branch[i], net.A_conv2[i][0], 1);
    CONNECT_BRANCH_I(net.A_branch[i], net.A_conv3[i][0], 2);
    CONNECT_BRANCH_I(net.A_branch[i], net.A_pool4[i], 3);

    CONNECT(net.A_conv1[i], net.A_bn1[i]);
    CONNECT(net.A_bn1[i], net.A_relu1[i]);

    for (int j = 0; j < 2; j++) {
      CONNECT(net.A_conv2[i][j], net.A_bn2[i][j]);
      CONNECT(net.A_bn2[i][j], net.A_relu2[i][j]);

      if (j != 1) {
        CONNECT(net.A_relu2[i][j], net.A_conv2[i][j+1]);
      }
    }

    for (int j = 0; j < 3; j++) {
      CONNECT(net.A_conv3[i][j], net.A_bn3[i][j]);
      CONNECT(net.A_bn3[i][j], net.A_relu3[i][j]);

      if (j != 2) {
        CONNECT(net.A_relu3[i][j], net.A_conv3[i][j+1]);
      }
    }

    CONNECT(net.A_pool4[i], net.A_conv4[i]);
    CONNECT(net.A_conv4[i], net.A_bn4[i]);
    CONNECT(net.A_bn4[i], net.A_relu4[i]);

    CONNECT_CONCAT_I(net.A_relu1[i], net.A_concat[i], 0);
    CONNECT_CONCAT_I(net.A_relu2[i][1], net.A_concat[i], 1);
    CONNECT_CONCAT_I(net.A_relu3[i][2], net.A_concat[i], 2);
    CONNECT_CONCAT_I(net.A_relu4[i], net.A_concat[i], 3);

    if (i != 2) {
      CONNECT(net.A_concat[i], net.A_branch[i+1]);
    }
  }

  CONNECT(net.A_concat[2], net.B_branch);

  {
    CONNECT_BRANCH_I(net.B_branch, net.B_pool1, 0);
    CONNECT_BRANCH_I(net.B_branch, net.B_conv2, 1);
    CONNECT_BRANCH_I(net.B_branch, net.B_conv3[0], 2);

    CONNECT(net.B_conv2, net.B_bn2);
    CONNECT(net.B_bn2, net.B_relu2);

    for (int j = 0; j < 3; j++) {
      CONNECT(net.B_conv3[j], net.B_bn3[j]);
      CONNECT(net.B_bn3[j], net.B_relu3[j]);

      if (j != 2) {
        CONNECT(net.B_relu3[j], net.B_conv3[j+1]);
      }
    }

    CONNECT_CONCAT_I(net.B_pool1, net.B_concat, 0);
    CONNECT_CONCAT_I(net.B_relu2, net.B_concat, 1);
    CONNECT_CONCAT_I(net.B_relu3[2], net.B_concat, 2);
  }

  CONNECT(net.B_concat, net.C_branch[0]);

  for (int i = 0; i < 4; i++) {
    CONNECT_BRANCH_I(net.C_branch[i], net.C_conv1[i], 0);
    CONNECT_BRANCH_I(net.C_branch[i], net.C_conv2[i][0], 1);
    CONNECT_BRANCH_I(net.C_branch[i], net.C_conv3[i][0], 2);
    CONNECT_BRANCH_I(net.C_branch[i], net.C_pool4[i], 3);

    CONNECT(net.C_conv1[i], net.C_bn1[i]);
    CONNECT(net.C_bn1[i], net.C_relu1[i]);

    for (int j = 0; j < 3; j++) {
      CONNECT(net.C_conv2[i][j], net.C_bn2[i][j]);
      CONNECT(net.C_bn2[i][j], net.C_relu2[i][j]);

      if (j != 2) {
        CONNECT(net.C_relu2[i][j], net.C_conv2[i][j+1]);
      }
    }

    for (int j = 0; j < 5; j++) {
      CONNECT(net.C_conv3[i][j], net.C_bn3[i][j]);
      CONNECT(net.C_bn3[i][j], net.C_relu3[i][j]);

      if (j != 4) {
        CONNECT(net.C_relu3[i][j], net.C_conv3[i][j+1]);
      }
    }

    CONNECT(net.C_pool4[i], net.C_conv4[i]);
    CONNECT(net.C_conv4[i], net.C_bn4[i]);
    CONNECT(net.C_bn4[i], net.C_relu4[i]);

    CONNECT_CONCAT_I(net.C_relu1[i], net.C_concat[i], 0);
    CONNECT_CONCAT_I(net.C_relu2[i][2], net.C_concat[i], 1);
    CONNECT_CONCAT_I(net.C_relu3[i][4], net.C_concat[i], 2);
    CONNECT_CONCAT_I(net.C_relu4[i], net.C_concat[i], 3);

    if (i != 3) {
      CONNECT(net.C_concat[i], net.C_branch[i+1]);
    }
  }

  CONNECT(net.C_concat[3], net.D_branch);

  {
    CONNECT_BRANCH_I(net.D_branch, net.D_pool1, 0);
    CONNECT_BRANCH_I(net.D_branch, net.D_conv2[0], 1);
    CONNECT_BRANCH_I(net.D_branch, net.D_conv3[0], 2);

    for (int j = 0; j < 2; j++) {
      CONNECT(net.D_conv2[j], net.D_bn2[j]);
      CONNECT(net.D_bn2[j], net.D_relu2[j]);

      if (j != 1) {
        CONNECT(net.D_relu2[j], net.D_conv2[j+1]);
      }
    }

    for (int j = 0; j < 4; j++) {
      CONNECT(net.D_conv3[j], net.D_bn3[j]);
      CONNECT(net.D_bn3[j], net.D_relu3[j]);

      if (j != 3) {
        CONNECT(net.D_relu3[j], net.D_conv3[j+1]);
      }
    }

    CONNECT_CONCAT_I(net.D_pool1, net.D_concat, 0);
    CONNECT_CONCAT_I(net.D_relu2[1], net.D_concat, 1);
    CONNECT_CONCAT_I(net.D_relu3[3], net.D_concat, 2);
  }

  CONNECT(net.D_concat, net.E_branch[0]);

  for (int i = 0; i < 2; i++) {
    CONNECT_BRANCH_I(net.E_branch[i], net.E_conv1[i], 0);
    CONNECT_BRANCH_I(net.E_branch[i], net.E_conv2[i][0], 1);
    CONNECT_BRANCH_I(net.E_branch[i], net.E_conv3[i][0], 2);
    CONNECT_BRANCH_I(net.E_branch[i], net.E_pool4[i], 3);

    CONNECT(net.E_conv1[i], net.E_bn1[i]);
    CONNECT(net.E_bn1[i], net.E_relu1[i]);

    CONNECT(net.E_conv2[i][0], net.E_bn2[i][0]);
    CONNECT(net.E_bn2[i][0], net.E_relu2[i][0]);
    CONNECT(net.E_relu2[i][0], net.E_branch2[i]);

    for (int j = 1; j < 3; j++) {
      CONNECT_BRANCH_I(net.E_branch2[i], net.E_conv2[i][j], (j-1));
      CONNECT(net.E_conv2[i][j], net.E_bn2[i][j]);
      CONNECT(net.E_bn2[i][j], net.E_relu2[i][j]);
    }

    CONNECT(net.E_conv3[i][0], net.E_bn3[i][0]);
    CONNECT(net.E_bn3[i][0], net.E_relu3[i][0]);
    CONNECT(net.E_relu3[i][0], net.E_conv3[i][1]);
    CONNECT(net.E_conv3[i][1], net.E_bn3[i][1]);
    CONNECT(net.E_bn3[i][1], net.E_relu3[i][1]);
    CONNECT(net.E_relu3[i][1], net.E_branch3[i]);

    for (int j = 2; j < 4; j++) {
      CONNECT_BRANCH_I(net.E_branch3[i], net.E_conv3[i][j], (j-2));
      CONNECT(net.E_conv3[i][j], net.E_bn3[i][j]);
      CONNECT(net.E_bn3[i][j], net.E_relu3[i][j]);
    }

    CONNECT(net.E_pool4[i], net.E_conv4[i]);
    CONNECT(net.E_conv4[i], net.E_bn4[i]);
    CONNECT(net.E_bn4[i], net.E_relu4[i]);

    CONNECT_CONCAT_I(net.E_relu1[i], net.E_concat[i], 0);
    CONNECT_CONCAT_I(net.E_relu2[i][1], net.E_concat[i], 1);
    CONNECT_CONCAT_I(net.E_relu2[i][2], net.E_concat[i], 2);
    CONNECT_CONCAT_I(net.E_relu3[i][2], net.E_concat[i], 3);
    CONNECT_CONCAT_I(net.E_relu3[i][3], net.E_concat[i], 4);
    CONNECT_CONCAT_I(net.E_relu4[i], net.E_concat[i], 5);

    if (i != 1) {
      CONNECT(net.E_concat[i], net.E_branch[i+1]);
    }
  }

  CONNECT(net.E_concat[1] , net.epilogue_pool);
  CONNECT(net.epilogue_pool, net.fc);
  CONNECT_DIRECT(net.fc, net.fc_bias, net.softmax);
}

void inception_clear_time()
{
  for (int i = 0; i < 3; i++) {
    clear_time_conv_layer(&net.prologue_conv[i]);
    clear_time_bn_layer(&net.prologue_bn[i]);
    clear_time_act_layer(&net.prologue_relu[i]);
  }

  clear_time_pool_layer(&net.prologue_pool[0]);

  for (int i = 3; i < 5; i++) {
    clear_time_conv_layer(&net.prologue_conv[i]);
    clear_time_bn_layer(&net.prologue_bn[i]);
    clear_time_act_layer(&net.prologue_relu[i]);
  }

  clear_time_pool_layer(&net.prologue_pool[1]);

  for (int i = 0; i < 3; i++) {
    clear_time_branch_layer(&net.A_branch[i]);
    clear_time_conv_layer(&net.A_conv1[i]);
    clear_time_bn_layer(&net.A_bn1[i]);
    clear_time_act_layer(&net.A_relu1[i]);

    for (int j = 0; j < 2; j++) {
      clear_time_conv_layer(&net.A_conv2[i][j]);
      clear_time_bn_layer(&net.A_bn2[i][j]);
      clear_time_act_layer(&net.A_relu2[i][j]);
    }

    for (int j = 0; j < 3; j++) {
      clear_time_conv_layer(&net.A_conv3[i][j]);
      clear_time_bn_layer(&net.A_bn3[i][j]);
      clear_time_act_layer(&net.A_relu3[i][j]);
    }

    clear_time_pool_layer(&net.A_pool4[i]);
    clear_time_conv_layer(&net.A_conv4[i]);
    clear_time_bn_layer(&net.A_bn4[i]);
    clear_time_act_layer(&net.A_relu4[i]);

    clear_time_concat_layer(&net.A_concat[i]);
  }

  {
    clear_time_branch_layer(&net.B_branch);

    clear_time_pool_layer(&net.B_pool1);
    clear_time_conv_layer(&net.B_conv2);
    clear_time_bn_layer(&net.B_bn2);
    clear_time_act_layer(&net.B_relu2);

    for (int j = 0; j < 3; j++) {
      clear_time_conv_layer(&net.B_conv3[j]);
      clear_time_bn_layer(&net.B_bn3[j]);
      clear_time_act_layer(&net.B_relu3[j]);
    }

    clear_time_concat_layer(&net.B_concat);
  }

  for (int i = 0; i < 4; i++) {
    clear_time_branch_layer(&net.C_branch[i]);
    clear_time_conv_layer(&net.C_conv1[i]);
    clear_time_bn_layer(&net.C_bn1[i]);
    clear_time_act_layer(&net.C_relu1[i]);

    for (int j = 0; j < 3; j++) {
      clear_time_conv_layer(&net.C_conv2[i][j]);
      clear_time_bn_layer(&net.C_bn2[i][j]);
      clear_time_act_layer(&net.C_relu2[i][j]);
    }

    for (int j = 0; j < 5; j++) {
      clear_time_conv_layer(&net.C_conv3[i][j]);
      clear_time_bn_layer(&net.C_bn3[i][j]);
      clear_time_act_layer(&net.C_relu3[i][j]);
    }

    clear_time_pool_layer(&net.C_pool4[i]);
    clear_time_conv_layer(&net.C_conv4[i]);
    clear_time_bn_layer(&net.C_bn4[i]);
    clear_time_act_layer(&net.C_relu4[i]);

    clear_time_concat_layer(&net.C_concat[i]);
  }

  {
    clear_time_branch_layer(&net.D_branch);
    clear_time_pool_layer(&net.D_pool1);

    for (int j = 0; j < 2; j++) {
      clear_time_conv_layer(&net.D_conv2[j]);
      clear_time_bn_layer(&net.D_bn2[j]);
      clear_time_act_layer(&net.D_relu2[j]);
    }

    for (int j = 0; j < 4; j++) {
      clear_time_conv_layer(&net.D_conv3[j]);
      clear_time_bn_layer(&net.D_bn3[j]);
      clear_time_act_layer(&net.D_relu3[j]);
    }

    clear_time_concat_layer(&net.D_concat);
  }  
  
  for (int i = 0; i < 2; i++) {
    clear_time_branch_layer(&net.E_branch[i]);
    clear_time_conv_layer(&net.E_conv1[i]);
    clear_time_bn_layer(&net.E_bn1[i]);
    clear_time_act_layer(&net.E_relu1[i]);

    for (int j = 0; j < 3; j++) {
      clear_time_conv_layer(&net.E_conv2[i][j]);
      clear_time_bn_layer(&net.E_bn2[i][j]);
      clear_time_act_layer(&net.E_relu2[i][j]);

      if (j == 0) {
        clear_time_branch_layer(&net.E_branch2[i]);
      }
    }

    for (int j = 0; j < 4; j++) {
      clear_time_conv_layer(&net.E_conv3[i][j]);
      clear_time_bn_layer(&net.E_bn3[i][j]);
      clear_time_act_layer(&net.E_relu3[i][j]);

      if (j == 1) {
        clear_time_branch_layer(&net.E_branch3[i]);
      }
    }

    clear_time_pool_layer(&net.E_pool4[i]);
    clear_time_conv_layer(&net.E_conv4[i]);
    clear_time_bn_layer(&net.E_bn4[i]);
    clear_time_act_layer(&net.E_relu4[i]);

    clear_time_concat_layer(&net.E_concat[i]);
  }

  clear_time_pool_layer(&net.epilogue_pool);
  clear_time_conv_layer(&net.fc);
  clear_time_bias_layer(&net.fc_bias);
  clear_time_softmax_layer(&net.softmax);
}

void inception_print_time()
{
  char buf[1024];
  printf("name, fwd, bwd_data, bwd_weight, update\n");

  for (int i = 0; i < 3; i++) {
    sprintf(buf, "prologue_conv[%d]", i);
    print_time_conv_layer(&net.prologue_conv[i], buf);
    sprintf(buf, "prologue_bn[%d]", i);
    print_time_bn_layer(&net.prologue_bn[i], buf);
    sprintf(buf, "prologue_relu[%d]", i);
    print_time_act_layer(&net.prologue_relu[i], buf);
  }

  sprintf(buf, "prologue_pool[%d]", 0);
  print_time_pool_layer(&net.prologue_pool[0], buf);

  for (int i = 3; i < 5; i++) {
    sprintf(buf, "prologue_conv[%d]", i);
    print_time_conv_layer(&net.prologue_conv[i], buf);
    sprintf(buf, "prologue_bn[%d]", i);
    print_time_bn_layer(&net.prologue_bn[i], buf);
    sprintf(buf, "prologue_relu[%d]", i);
    print_time_act_layer(&net.prologue_relu[i], buf);
  }

  sprintf(buf, "prologue_pool[%d]", 1);
  print_time_pool_layer(&net.prologue_pool[1], buf);

  for (int i = 0; i < 3; i++) {
    sprintf(buf, "A_branch[%d]", i);
    print_time_branch_layer(&net.A_branch[i], buf);
    sprintf(buf, "A_conv1[%d]", i);
    print_time_conv_layer(&net.A_conv1[i], buf);
    sprintf(buf, "A_bn1[%d]", i);
    print_time_bn_layer(&net.A_bn1[i], buf);
    sprintf(buf, "A_relu1[%d]", i);
    print_time_act_layer(&net.A_relu1[i], buf);

    for (int j = 0; j < 2; j++) {
      sprintf(buf, "A_conv2[%d][%d]", i, j);
      print_time_conv_layer(&net.A_conv2[i][j], buf);
      sprintf(buf, "A_bn2[%d][%d]", i, j);
      print_time_bn_layer(&net.A_bn2[i][j], buf);
      sprintf(buf, "A_relu2[%d][%d]", i, j);
      print_time_act_layer(&net.A_relu2[i][j], buf);
    }

    for (int j = 0; j < 3; j++) {
      sprintf(buf, "A_conv3[%d][%d]", i, j);
      print_time_conv_layer(&net.A_conv3[i][j], buf);
      sprintf(buf, "A_bn3[%d][%d]", i, j);
      print_time_bn_layer(&net.A_bn3[i][j], buf);
      sprintf(buf, "A_relu3[%d][%d]", i, j);
      print_time_act_layer(&net.A_relu3[i][j], buf);
    }

    sprintf(buf, "A_pool4[%d]", i);
    print_time_pool_layer(&net.A_pool4[i], buf);
    sprintf(buf, "A_conv4[%d]", i);
    print_time_conv_layer(&net.A_conv4[i], buf);
    sprintf(buf, "A_bn4[%d]", i);
    print_time_bn_layer(&net.A_bn4[i], buf);
    sprintf(buf, "A_relu4[%d]", i);
    print_time_act_layer(&net.A_relu4[i], buf);

    sprintf(buf, "A_concat[%d]", i);
    print_time_concat_layer(&net.A_concat[i], buf);
  }

  {
    sprintf(buf, "B_branch");
    print_time_branch_layer(&net.B_branch, buf);

    sprintf(buf, "B_pool1");
    print_time_pool_layer(&net.B_pool1, buf);
    sprintf(buf, "B_conv2");
    print_time_conv_layer(&net.B_conv2, buf);
    sprintf(buf, "B_bn2");
    print_time_bn_layer(&net.B_bn2, buf);
    sprintf(buf, "B_relu2");
    print_time_act_layer(&net.B_relu2, buf);

    for (int j = 0; j < 3; j++) {
      sprintf(buf, "B_conv3[%d]", j);
      print_time_conv_layer(&net.B_conv3[j], buf);
      sprintf(buf, "B_bn3[%d]", j);
      print_time_bn_layer(&net.B_bn3[j], buf);
      sprintf(buf, "B_relu3[%d]", j);
      print_time_act_layer(&net.B_relu3[j], buf);
    }

    sprintf(buf, "B_concat");
    print_time_concat_layer(&net.B_concat, buf);
  }

  for (int i = 0; i < 4; i++) {
    sprintf(buf, "C_branch[%d]", i);
    print_time_branch_layer(&net.C_branch[i], buf);
    sprintf(buf, "C_conv1[%d]", i);
    print_time_conv_layer(&net.C_conv1[i], buf);
    sprintf(buf, "C_bn1[%d]", i);
    print_time_bn_layer(&net.C_bn1[i], buf);
    sprintf(buf, "C_relu1[%d]", i);
    print_time_act_layer(&net.C_relu1[i], buf);

    for (int j = 0; j < 3; j++) {
      sprintf(buf, "C_conv2[%d][%d]", i, j);
      print_time_conv_layer(&net.C_conv2[i][j], buf);
      sprintf(buf, "C_bn2[%d][%d]", i, j);
      print_time_bn_layer(&net.C_bn2[i][j], buf);
      sprintf(buf, "C_relu2[%d][%d]", i, j);
      print_time_act_layer(&net.C_relu2[i][j], buf);
    }

    for (int j = 0; j < 5; j++) {
      sprintf(buf, "C_conv3[%d][%d]", i, j);
      print_time_conv_layer(&net.C_conv3[i][j], buf);
      sprintf(buf, "C_bn3[%d][%d]", i, j);
      print_time_bn_layer(&net.C_bn3[i][j], buf);
      sprintf(buf, "C_relu3[%d][%d]", i, j);
      print_time_act_layer(&net.C_relu3[i][j], buf);
    }

    sprintf(buf, "C_pool4[%d]", i);
    print_time_pool_layer(&net.C_pool4[i], buf);
    sprintf(buf, "C_conv4[%d]", i);
    print_time_conv_layer(&net.C_conv4[i], buf);
    sprintf(buf, "C_bn4[%d]", i);
    print_time_bn_layer(&net.C_bn4[i], buf);
    sprintf(buf, "C_relu4[%d]", i);
    print_time_act_layer(&net.C_relu4[i], buf);

    sprintf(buf, "C_concat[%d]", i);
    print_time_concat_layer(&net.C_concat[i], buf);
  }

  {
    sprintf(buf, "D_branch");
    print_time_branch_layer(&net.D_branch, buf);
    sprintf(buf, "D_pool1");
    print_time_pool_layer(&net.D_pool1, buf);

    for (int j = 0; j < 2; j++) {
      sprintf(buf, "D_conv2[%d]", j);
      print_time_conv_layer(&net.D_conv2[j], buf);
      sprintf(buf, "D_bn2[%d]", j);
      print_time_bn_layer(&net.D_bn2[j], buf);
      sprintf(buf, "D_relu2[%d]", j);
      print_time_act_layer(&net.D_relu2[j], buf);
    }

    for (int j = 0; j < 4; j++) {
      sprintf(buf, "D_conv3[%d]", j);
      print_time_conv_layer(&net.D_conv3[j], buf);
      sprintf(buf, "D_bn3[%d]", j);
      print_time_bn_layer(&net.D_bn3[j], buf);
      sprintf(buf, "D_relu3[%d]", j);
      print_time_act_layer(&net.D_relu3[j], buf);
    }

    sprintf(buf, "D_concat");
    print_time_concat_layer(&net.D_concat, buf);
  }  
  
  for (int i = 0; i < 2; i++) {
    sprintf(buf, "E_branch[%d]", i);
    print_time_branch_layer(&net.E_branch[i], buf);
    sprintf(buf, "E_conv1[%d]", i);
    print_time_conv_layer(&net.E_conv1[i], buf);
    sprintf(buf, "E_bn1[%d]", i);
    print_time_bn_layer(&net.E_bn1[i], buf);
    sprintf(buf, "E_relu1[%d]", i);
    print_time_act_layer(&net.E_relu1[i], buf);

    for (int j = 0; j < 3; j++) {
      sprintf(buf, "E_conv2[%d][%d]", i, j);
      print_time_conv_layer(&net.E_conv2[i][j], buf);
      sprintf(buf, "E_bn2[%d][%d]", i, j);
      print_time_bn_layer(&net.E_bn2[i][j], buf);
      sprintf(buf, "E_relu2[%d][%d]", i, j);
      print_time_act_layer(&net.E_relu2[i][j], buf);

      if (j == 0) {
        sprintf(buf, "E_branch2[%d]", i);
        print_time_branch_layer(&net.E_branch2[i], buf);
      }
    }

    for (int j = 0; j < 4; j++) {
      sprintf(buf, "E_conv3[%d][%d]", i, j);
      print_time_conv_layer(&net.E_conv3[i][j], buf);
      sprintf(buf, "E_bn3[%d][%d]", i, j);
      print_time_bn_layer(&net.E_bn3[i][j], buf);
      sprintf(buf, "E_relu3[%d][%d]", i, j);
      print_time_act_layer(&net.E_relu3[i][j], buf);

      if (j == 1) {
        sprintf(buf, "E_branch3[%d]", i);
        print_time_branch_layer(&net.E_branch3[i], buf);
      }
    }

    sprintf(buf, "E_pool4[%d]", i);
    print_time_pool_layer(&net.E_pool4[i], buf);
    sprintf(buf, "E_conv4[%d]", i);
    print_time_conv_layer(&net.E_conv4[i], buf);
    sprintf(buf, "E_bn4[%d]", i);
    print_time_bn_layer(&net.E_bn4[i], buf);
    sprintf(buf, "E_relu4[%d]", i);
    print_time_act_layer(&net.E_relu4[i], buf);

    sprintf(buf, "E_concat[%d]", i);
    print_time_concat_layer(&net.E_concat[i], buf);
  }

  sprintf(buf, "epilogue_pool");
  print_time_pool_layer(&net.epilogue_pool, buf);
  sprintf(buf, "fc");
  print_time_conv_layer(&net.fc, buf);
  sprintf(buf, "fc_bias");
  print_time_bias_layer(&net.fc_bias, buf);
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

void cnn_train(
  int num_train_image,
  float * train_data, int * train_label) 
{
  assert(num_train_image % params.batch_size == 0); 

  inception_init(params.batch_size);
  inception_connect();

  int num_batches = num_train_image / params.batch_size;
  fprintf(stderr, "total iteration : %d\n", num_batches);

  size_t sz = inception_get_param_size();
  float *param_in = (float *)malloc(sz);
  float *param_out = (float *)malloc(sz);
  float *param_result = (float *)malloc(sz);
  INITIALIZE_RAND(param_in, sz/sizeof(float));

  inception_set_param(param_in);

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

      inception_copy_input(batch_size, data_in, label_in);

      inception_forward();

#ifdef PRINT_LOSS
      float l = get_loss(&net.softmax, label_in);
      printf("loss for %d/%d : %f\n", b, num_batches, l);
#endif

      inception_backward();

      if (first) {
        cudaDeviceSynchronize();
        clock_gettime(CLOCK_MONOTONIC, &ed_f);
#ifdef TIME_LAYER
        inception_clear_time();
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
  inception_print_time();
#endif

  inception_get_param(param_out);

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

