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

typedef struct vgg_s {
  input_layer input;
  conv_layer conv[13];
  bias_layer bias[13];
  act_layer relu[13];
  pool_layer pool[5];
  fc_layer fc[3];
  bias_layer fc_bias[3];
  act_layer fc_relu[2];
  softmax_layer softmax;
} vgg;

vgg net;

bool is_initiated = false;

void params_modify()
{
}

void vgg_init(int batch_size)
{
  chkCUDNN(cudnnCreate(&cudnn));
  chkCUBLAS(cublasCreate(&cublas));
  srand(params.seed);

  init_input_layer(&net.input, cudnn, batch_size, 3, 224, 224);
  int prev_channel = 3;

  for (int i = 0; i < 2; i++) {
    init_conv_layer(&net.conv[i], cudnn, batch_size, 3, 3, 1, 1, 1, 1, prev_channel, 64, 224, 224);
    init_bias_layer(&net.bias[i], cudnn, batch_size, 64, 224, 224);
    init_act_layer(&net.relu[i], cudnn, batch_size, 64, 224, 224);
    prev_channel = 64;
  }

  init_pool_layer(&net.pool[0], cudnn, batch_size, 2, 2, 0, 0, 2, 2, 64, 224, 224, max);

  for (int i = 2; i < 4; i++) {
    init_conv_layer(&net.conv[i], cudnn, batch_size, 3, 3, 1, 1, 1, 1, prev_channel, 128, 112, 112);
    init_bias_layer(&net.bias[i], cudnn, batch_size, 128, 112, 112);
    init_act_layer(&net.relu[i], cudnn, batch_size, 128, 112, 112);
    prev_channel = 128;
  }

  init_pool_layer(&net.pool[1], cudnn, batch_size, 2, 2, 0, 0, 2, 2, 128, 112, 112, max);

  for (int i = 4; i < 7; i++) {
    init_conv_layer(&net.conv[i], cudnn, batch_size, 3, 3, 1, 1, 1, 1, prev_channel, 256, 56, 56);
    init_bias_layer(&net.bias[i], cudnn, batch_size, 256, 56, 56);
    init_act_layer(&net.relu[i], cudnn, batch_size, 256, 56, 56);
    prev_channel = 256;
  }

  init_pool_layer(&net.pool[2], cudnn, batch_size, 2, 2, 0, 0, 2, 2, 256, 56, 56, max);

  for (int i = 7; i < 10; i++) {
    init_conv_layer(&net.conv[i], cudnn, batch_size, 3, 3, 1, 1, 1, 1, prev_channel, 512, 28, 28);
    init_bias_layer(&net.bias[i], cudnn, batch_size, 512, 28, 28);
    init_act_layer(&net.relu[i], cudnn, batch_size, 512, 28, 28);
    prev_channel = 512;
  }

  init_pool_layer(&net.pool[3], cudnn, batch_size, 2, 2, 0, 0, 2, 2, 512, 28, 28, max);

  for (int i = 10; i < 13; i++) {
    init_conv_layer(&net.conv[i], cudnn, batch_size, 3, 3, 1, 1, 1, 1, prev_channel, 512, 14, 14);
    init_bias_layer(&net.bias[i], cudnn, batch_size, 512, 14, 14);
    init_act_layer(&net.relu[i], cudnn, batch_size, 512, 14, 14);
    prev_channel = 512;
  }

  init_pool_layer(&net.pool[4], cudnn, batch_size, 2, 2, 0, 0, 2, 2, 512, 14, 14, max);
  prev_channel = 7 * 7 * 512;

  for (int i = 0; i < 2; i++) {
    init_fc_layer(&net.fc[i], cudnn, batch_size, prev_channel, 4096);
    init_bias_layer(&net.fc_bias[i], cudnn, batch_size, 4096, 1, 1);
    init_act_layer(&net.fc_relu[i], cudnn, batch_size, 4096, 1, 1);
    prev_channel = 4096;
  }

  init_fc_layer(&net.fc[2], cudnn, batch_size, prev_channel, 1000);
  init_bias_layer(&net.fc_bias[2], cudnn, batch_size, 1000, 1, 1);
  init_softmax_layer(&net.softmax, cudnn, batch_size, 1000);

  init_conv_workspace();

  is_initiated = true;
}

size_t vgg_get_param_size()
{
  size_t sum = 0;

  for (int i = 0; i < 13; i++) {
    sum += PSIZE_CONV(net.conv[i]);
    sum += PSIZE_BIAS(net.bias[i]);
  }

  for (int i = 0; i < 3; i++) {
    sum += PSIZE_FC(net.fc[i]);
    sum += PSIZE_BIAS(net.fc_bias[i]);
  }

  return sum;
}

void vgg_load_param(float *param)
{
  for (int i = 0; i < 13; i++) {
    LOAD_CONV(net.conv[i]);
    LOAD_BIAS(net.bias[i]);
  }

  for (int i = 0; i < 3; i++) {
    LOAD_FC(net.fc[i]);
    LOAD_BIAS(net.fc_bias[i]);
  }
}

void vgg_set_param(float *param)
{
  for (int i = 0; i < 13; i++) {
    INIT_CONV(net.conv[i]);
    INIT_BIAS(net.bias[i]);
  }

  for (int i = 0; i < 3; i++) {
    INIT_FC(net.fc[i]);
    INIT_BIAS(net.fc_bias[i]);
  }
}

void vgg_get_param(float *param)
{
  for (int i = 0; i < 13; i++) {
    param += get_conv_filter(net.conv[i], param);
    param += get_bias(net.bias[i], param);
  }

  for (int i = 0; i < 3; i++) {
    param += get_fc_weight(net.fc[i], param);
    param += get_bias(net.fc_bias[i], param);
  }
}

void vgg_copy_input(int batch_size, float *data_in, int *label_in)
{
  size_t input_size = sizeof(float) * batch_size * params.width * params.height * params.channel;

  chkCUDA(cudaMemcpy(net.input.output, data_in, input_size, cudaMemcpyHostToDevice)); 
  chkCUDA(cudaMemcpy(net.softmax.label_in, label_in, batch_size * sizeof(int), cudaMemcpyHostToDevice)); 
  cuda_set_label(batch_size, 1000, net.softmax.label_in, net.softmax.label);
}

void vgg_forward()
{
  for (int i = 0, j = 0; i < 13; i++) {
    train_fwd_conv_layer(&net.conv[i]);
    train_fwd_bias_layer(&net.bias[i]);
    train_fwd_act_layer(&net.relu[i]);

    if (i == 1 || i == 3 || i == 6 || i == 9 || i == 12) {
      train_fwd_pool_layer(&net.pool[j]);
      j++;
    }
  }

  for (int i = 0; i < 2; i++) {
    train_fwd_fc_layer(&net.fc[i]);
    train_fwd_bias_layer(&net.fc_bias[i]);
    train_fwd_act_layer(&net.fc_relu[i]);
  }

  train_fwd_fc_layer(&net.fc[2]);
  train_fwd_bias_layer(&net.fc_bias[2]);
  train_fwd_softmax_layer(&net.softmax);
}

void vgg_backward()
{
  train_bwd_softmax_layer(&net.softmax);
  train_bwd_bias_layer(&net.fc_bias[2]);
  train_bwd_fc_layer(&net.fc[2]);

  for (int i = 1; i >= 0; i--) {
    train_bwd_act_layer(&net.fc_relu[i]);
    train_bwd_bias_layer(&net.fc_bias[i]);
    train_bwd_fc_layer(&net.fc[i]);
  }

  for (int i = 12, j = 4; i >= 0; i--) {
    if (i == 1 || i == 3 || i == 6 || i == 9 || i == 12) {
      train_bwd_pool_layer(&net.pool[j]);
      j--;
    }

    train_bwd_act_layer(&net.relu[i]);
    train_bwd_bias_layer(&net.bias[i]);
    train_bwd_conv_layer(&net.conv[i]);
  }
}

void vgg_connect()
{
  for (int i = 0, j = 0; i < 13; i++) {
    if (i == 0) {
      CONNECT(net.input, net.conv[i]);
    }
    else if (i == 2 || i == 4 || i == 7 || i == 10 || i == 13) {
      CONNECT(net.pool[j], net.conv[i]);
      j++;
    }
    else {
      CONNECT(net.relu[i-1], net.conv[i]);
    }

    CONNECT_DIRECT(net.conv[i], net.bias[i], net.relu[i]);

    if (i == 1 || i == 3 || i == 6 || i == 9 || i == 12) {
      CONNECT(net.relu[i], net.pool[j]);
    }
  }

  CONNECT(net.pool[4], net.fc[0]);

  for (int i = 0; i < 2; i++) {
    CONNECT_DIRECT(net.fc[i], net.fc_bias[i], net.fc_relu[i]);
    CONNECT(net.fc_relu[i], net.fc[i+1]);
  }

  CONNECT_DIRECT(net.fc[2], net.fc_bias[2], net.softmax);
}

void vgg_clear_time()
{
  for (int i = 0, j = 0; i < 13; i++) {
    clear_time_conv_layer(&net.conv[i]);
    clear_time_bias_layer(&net.bias[i]);
    clear_time_act_layer(&net.relu[i]);

    if (i == 1 || i == 3 || i == 6 || i == 9 || i == 12) {
      clear_time_pool_layer(&net.pool[j]);
      j++;
    }
  }

  for (int i = 0; i < 3; i++) {
    clear_time_fc_layer(&net.fc[i]);
    clear_time_bias_layer(&net.fc_bias[i]);

    if (i < 2) {
      clear_time_act_layer(&net.fc_relu[i]);
    }
  }

  clear_time_softmax_layer(&net.softmax);
}

void vgg_print_time()
{
  char buf[1024];
  printf("name, fwd, bwd_data, bwd_weight, update\n");

  for (int i = 0, j = 0; i < 13; i++) {
    sprintf(buf, "conv%d", i);
    print_time_conv_layer(&net.conv[i], buf);
    sprintf(buf, "bias%d", i);
    print_time_bias_layer(&net.bias[i], buf);
    sprintf(buf, "relu%d", i);
    print_time_act_layer(&net.relu[i], buf);

    if (i == 1 || i == 3 || i == 6 || i == 9 || i == 12) {
      sprintf(buf, "pool%d", i);
      print_time_pool_layer(&net.pool[j], buf);
      j++;
    }
  }

  for (int i = 0; i < 3; i++) {
    sprintf(buf, "fc%d", i);
    print_time_fc_layer(&net.fc[i], buf);
    sprintf(buf, "fc_bias%d", i);
    print_time_bias_layer(&net.fc_bias[i], buf);

    if (i < 2) {
      sprintf(buf, "fc_relu%d", i);
      print_time_act_layer(&net.fc_relu[i], buf);
    }
  }

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

void vgg_dump(FILE *f)
{
  char buf[1024];

  for (int i = 0, j = 0; i < 13; i++) {
    sprintf(buf, "conv%d", i);
    DUMP_BOTH(net.conv[i], buf, f);
    sprintf(buf, "relu%d", i);
    DUMP_BOTH(net.relu[i], buf, f);

    if (i == 1 || i == 3 || i == 6 || i == 9 || i == 12) {
      sprintf(buf, "pool%d", i);
      DUMP_BOTH(net.pool[j], buf, f);
      j++;
    }
  }

  for (int i = 0; i < 3; i++) {
    sprintf(buf, "fc%d", i);
    DUMP_BOTH(net.fc[i], buf, f);

    if (i < 2) {
      sprintf(buf, "fc_relu%d", i);
      DUMP_BOTH(net.fc_relu[i], buf, f);
    }
  }

  sprintf(buf, "softmax");
  DUMP_BOTH(net.softmax, buf, f);
}

void vgg_check(FILE *f)
{
  char buf[1024];

  for (int i = 0, j = 0; i < 13; i++) {
    sprintf(buf, "conv%d", i);
    LOAD_AND_CHECK_BOTH( net.conv[i], buf, f);
    sprintf(buf, "relu%d", i);
    LOAD_AND_CHECK_BOTH( net.relu[i], buf, f);

    if (i == 1 || i == 3 || i == 6 || i == 9 || i == 12) {
      sprintf(buf, "pool%d", i);
      LOAD_AND_CHECK_BOTH( net.pool[j], buf, f);
      j++;
    }
  }

  for (int i = 0; i < 3; i++) {
    sprintf(buf, "fc%d", i);
    LOAD_AND_CHECK_BOTH( net.fc[i], buf, f);
    if (i < 2) {
      sprintf(buf, "fc_relu%d", i);
      LOAD_AND_CHECK_BOTH( net.fc_relu[i], buf, f);
    }
  }

  sprintf(buf, "softmax");
  LOAD_AND_CHECK_BOTH( net.softmax, buf, f);
}

void cnn_train(int num_train_image, float *train_data, int *train_label) 
{
  assert(num_train_image % params.batch_size == 0); 

  vgg_init(params.batch_size);
  vgg_connect();

  int num_batches = num_train_image / params.batch_size;
  fprintf(stderr, "total iteration : %d\n", num_batches);

  size_t sz = vgg_get_param_size();
  float *param_in = (float *)malloc(sz);
  float *param_out = (float *)malloc(sz);
  float *param_result = (float *)malloc(sz);
  INITIALIZE_RAND(param_in, sz / sizeof(float));

  vgg_set_param(param_in);

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

      vgg_copy_input(batch_size, data_in, label_in);

      vgg_forward();

#ifdef PRINT_LOSS
      float l = get_loss(&net.softmax, label_in);
      printf("loss for %d/%d : %f\n", b, num_batches, l);
#endif

      vgg_backward();

      if (first) {
        cudaDeviceSynchronize();
        clock_gettime(CLOCK_MONOTONIC, &ed_f);
#ifdef TIME_LAYER
        vgg_clear_time();
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
  vgg_print_time();
#endif

  vgg_get_param(param_out);

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
#ifdef CHK_OUTPUT
  if (exists(params.result_output)) {
    FILE *f = fopen(params.result_output, "rb");
    vgg_check(f);
    fclose(f);
  }
  else {
    FILE *f = fopen(params.result_output, "wb");
    vgg_dump(f);
    fclose(f);
  }
#endif
}
