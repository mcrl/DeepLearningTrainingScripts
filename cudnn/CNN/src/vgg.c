#include <stdio.h>
#include <string.h>
#include <stdbool.h>
#include <assert.h>

#include "cnn.h"
#include "params.h"
#include "layer.h"
#include "utils.h"
#include "execute.h"

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
  bool is_initiated;
} vgg;

vgg net;

void params_modify()
{
}

void vgg_init(int batch_size)
{
  char name[1024];

  srand(params.seed);

  sprintf(name, "input");
  init_input_layer(&net.input, name, batch_size, 3, 224, 224);

  for (int i = 0; i < 2; i++) {
    sprintf(name, "conv_%02d", i);
    init_conv_layer(&net.conv[i], name, batch_size, 3, 3, 1, 1, 1, 1, 3, 64, 224, 224);

    sprintf(name, "bias_%02d", i);
    init_bias_layer(&net.bias[i], name, batch_size, 64, 224, 224);

    sprintf(name, "relu_%02d", i);
    init_act_layer(&net.relu[i], name, batch_size, 64, 224, 224, RELU_T);
  }

  sprintf(name, "pool_0");
  init_pool_layer(&net.pool[0], name, batch_size, 2, 2, 0, 0, 2, 2, 64, 224, 224, MAX_T);

  for (int i = 2; i < 4; i++) {
    sprintf(name, "conv_%02d", i);
    init_conv_layer(&net.conv[i], name, batch_size, 3, 3, 1, 1, 1, 1, 64, 128, 112, 112);

    sprintf(name, "bias_%02d", i);
    init_bias_layer(&net.bias[i], name, batch_size, 128, 112, 112);

    sprintf(name, "relu_%02d", i);
    init_act_layer(&net.relu[i], name, batch_size, 128, 112, 112, RELU_T);
  }

  sprintf(name, "pool_1");
  init_pool_layer(&net.pool[1], name, batch_size, 2, 2, 0, 0, 2, 2, 128, 112, 112, MAX_T);

  for (int i = 4; i < 7; i++) {
    sprintf(name, "conv_%02d", i);
    init_conv_layer(&net.conv[i], name, batch_size, 3, 3, 1, 1, 1, 1, 128, 256, 56, 56);

    sprintf(name, "bias_%02d", i);
    init_bias_layer(&net.bias[i], name, batch_size, 256, 56, 56);

    sprintf(name, "relu_%02d", i);
    init_act_layer(&net.relu[i], name, batch_size, 256, 56, 56, RELU_T);
  }

  sprintf(name, "pool_2");
  init_pool_layer(&net.pool[2], name, batch_size, 2, 2, 0, 0, 2, 2, 256, 56, 56, MAX_T);

  for (int i = 7; i < 10; i++) {
    sprintf(name, "conv_%02d", i);
    init_conv_layer(&net.conv[i], name, batch_size, 3, 3, 1, 1, 1, 1, 256, 512, 28, 28);

    sprintf(name, "bias_%02d", i);
    init_bias_layer(&net.bias[i], name, batch_size, 512, 28, 28);

    sprintf(name, "relu_%02d", i);
    init_act_layer(&net.relu[i], name, batch_size, 512, 28, 28, RELU_T);
  }

  sprintf(name, "pool_3");
  init_pool_layer(&net.pool[3], name, batch_size, 2, 2, 0, 0, 2, 2, 512, 28, 28, MAX_T);

  for (int i = 10; i < 13; i++) {
    sprintf(name, "conv_%02d", i);
    init_conv_layer(&net.conv[i], name, batch_size, 3, 3, 1, 1, 1, 1, 512, 512, 14, 14);

    sprintf(name, "bias_%02d", i);
    init_bias_layer(&net.bias[i], name, batch_size, 512, 14, 14);

    sprintf(name, "relu_%02d", i);
    init_act_layer(&net.relu[i], name, batch_size, 512, 14, 14, RELU_T);
  }

  sprintf(name, "pool_4");
  init_pool_layer(&net.pool[4], name, batch_size, 2, 2, 0, 0, 2, 2, 512, 14, 14, MAX_T);

  for (int i = 0; i < 2; i++) {
    sprintf(name, "fc_%d", i);
    init_fc_layer(&net.fc[i], name, batch_size, 512 * 7 * 7, 4096);

    sprintf(name, "fc_bias_%d", i);
    init_bias_layer(&net.fc_bias[i], name, batch_size, 4096, 1, 1);

    sprintf(name, "fc_relu_%d", i);
    init_act_layer(&net.fc_relu[i], name, batch_size, 4096, 1, 1, RELU_T);
  }

  sprintf(name, "fc_2");
  init_fc_layer(&net.fc[2], name, batch_size, 4096, 1000);

  sprintf(name, "fc_bias_2");
  init_bias_layer(&net.fc_bias[2], name, batch_size, 1000, 1, 1);

  sprintf(name, "softmax");
  init_softmax_layer(&net.softmax, name, batch_size, 1000);

  net.is_initiated = true;
}

#define VGG_PARAM(FUNC) \
do {\
  for (int i = 0; i < 13; i++) {\
    FUNC##_CONV(&net.conv[i]);\
    FUNC##_BIAS(&net.bias[i]);\
  }\
  for (int i = 0; i < 3; i++) {\
    FUNC##_FC(&net.fc[i]);\
    FUNC##_BIAS(&net.fc_bias[i]);\
  }\
} while (0)

size_t vgg_get_param_size()
{
  size_t sum = 0;

  VGG_PARAM(SIZE);

  return sum;
}

void vgg_load_param(float *param)
{
  VGG_PARAM(LOAD);
}

void vgg_init_param(float *param)
{
  VGG_PARAM(INIT);
}

void vgg_get_param(float *param)
{
  VGG_PARAM(GET);
}

void vgg_copy_input(float *data_in, int *label_in)
{
  set_input(&net.input, data_in);
  set_label(&net.softmax, label_in);
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
  CONNECT_INPUT(net.input);

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

    CONNECT_BIAS(net.conv[i], net.bias[i], net.relu[i]);

    if (i == 1 || i == 3 || i == 6 || i == 9 || i == 12) {
      CONNECT(net.relu[i], net.pool[j]);
    }
  }

  CONNECT(net.pool[4], net.fc[0]);

  for (int i = 0; i < 2; i++) {
    CONNECT_BIAS(net.fc[i], net.fc_bias[i], net.fc_relu[i]);
    CONNECT(net.fc_relu[i], net.fc[i+1]);
  }

  CONNECT_BIAS(net.fc[2], net.fc_bias[2], net.softmax);
}

#define VGG_LAYER(FUNC) \
do {\
  for (int i = 0, j = 0; i < 13; i++) {\
    FUNC##_conv_layer(&net.conv[i]);\
    FUNC##_bias_layer(&net.bias[i]);\
    FUNC##_act_layer(&net.relu[i]);\
    if (i == 1 || i == 3 || i == 6 || i == 9 || i == 12) {\
      FUNC##_pool_layer(&net.pool[j]);\
      j++;\
    }\
  }\
  for (int i = 0; i < 3; i++) {\
    FUNC##_fc_layer(&net.fc[i]);\
    FUNC##_bias_layer(&net.fc_bias[i]);\
    if (i < 2) {\
      FUNC##_act_layer(&net.fc_relu[i]);\
    }\
  }\
  FUNC##_softmax_layer(&net.softmax);\
} while (0)

void vgg_clear_time()
{
  VGG_LAYER(clear_time);
}

void vgg_print_time()
{
  printf("name, fwd, bwd_data, bwd_weight, update\n");

  VGG_LAYER(print_time);
}

void cnn_train(int num_train_image, float *train_data, int *train_label)
{
  assert(num_train_image % params.batch_size == 0);

  __init_stream_executer();
  __init_object_manager();

  printf("init layers\n");
  vgg_init(params.batch_size);
  printf("connect layers\n");
  vgg_connect();
  printf("alloc workspace\n");
  alloc_work_space();

  int num_batches = num_train_image / params.batch_size;
  fprintf(stderr, "total iteration : %d\n", num_batches);

  size_t sz = vgg_get_param_size();
  float *param_in = (float *)malloc(sz);
  float *param_out = (float *)malloc(sz);
  float *param_result = (float *)malloc(sz);

  INITIALIZE_RAND(param_in, sz / sizeof(float));

  printf("init params\n");
  vgg_init_param(param_in);

  struct timespec st;
  struct timespec st_f;
  struct timespec ed;
  struct timespec ed_f;

  int first = 1;

  printf("timer start\n");

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

      printf("[%d] copy input\n", b);
      vgg_copy_input(data_in, label_in);

      printf("[%d] forward\n", b);
      vgg_forward();

#ifdef PRINT_LOSS
      float l = get_loss(&net.softmax, label_in);
      printf("loss for %d/%d : %f\n", b, num_batches, l);
#endif

      printf("[%d] backward\n", b);
      vgg_backward();

      if (first) {
        cudaDeviceSynchronize(); // FIXME
        clock_gettime(CLOCK_MONOTONIC, &ed_f);
#ifdef TIME_LAYER
        vgg_clear_time();
#endif
        first = 0;
      }
    }
  }

  cudaDeviceSynchronize(); // FIXME
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
}
