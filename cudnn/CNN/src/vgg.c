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

vgg net = {
  .is_initiated = false
};

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
    sprintf(name, "conv[%02d]", i);
    if (i == 0) {
      init_conv_layer(&net.conv[i], name, batch_size, 3, 3, 1, 1, 1, 1, 3, 64, 224, 224);
    }
    else {
      init_conv_layer(&net.conv[i], name, batch_size, 3, 3, 1, 1, 1, 1, 64, 64, 224, 224);
    }

    sprintf(name, "bias[%02d]", i);
    init_bias_layer(&net.bias[i], name, batch_size, 64, 224, 224);

    sprintf(name, "relu[%02d]", i);
    init_act_layer(&net.relu[i], name, batch_size, 64, 224, 224, RELU_T);
  }

  sprintf(name, "pool[0]");
  init_pool_layer(&net.pool[0], name, batch_size, 2, 2, 0, 0, 2, 2, 64, 224, 224, MAX_T);

  for (int i = 2; i < 4; i++) {
    sprintf(name, "conv[%02d]", i);
    if (i == 2) {
      init_conv_layer(&net.conv[i], name, batch_size, 3, 3, 1, 1, 1, 1, 64, 128, 112, 112);
    }
    else {
      init_conv_layer(&net.conv[i], name, batch_size, 3, 3, 1, 1, 1, 1, 128, 128, 112, 112);
    }

    sprintf(name, "bias[%02d]", i);
    init_bias_layer(&net.bias[i], name, batch_size, 128, 112, 112);

    sprintf(name, "relu[%02d]", i);
    init_act_layer(&net.relu[i], name, batch_size, 128, 112, 112, RELU_T);
  }

  sprintf(name, "pool[1]");
  init_pool_layer(&net.pool[1], name, batch_size, 2, 2, 0, 0, 2, 2, 128, 112, 112, MAX_T);

  for (int i = 4; i < 7; i++) {
    sprintf(name, "conv[%02d]", i);
    if (i == 4) {
      init_conv_layer(&net.conv[i], name, batch_size, 3, 3, 1, 1, 1, 1, 128, 256, 56, 56);
    }
    else {
      init_conv_layer(&net.conv[i], name, batch_size, 3, 3, 1, 1, 1, 1, 256, 256, 56, 56);
    }

    sprintf(name, "bias[%02d]", i);
    init_bias_layer(&net.bias[i], name, batch_size, 256, 56, 56);

    sprintf(name, "relu[%02d]", i);
    init_act_layer(&net.relu[i], name, batch_size, 256, 56, 56, RELU_T);
  }

  sprintf(name, "pool[2]");
  init_pool_layer(&net.pool[2], name, batch_size, 2, 2, 0, 0, 2, 2, 256, 56, 56, MAX_T);

  for (int i = 7; i < 10; i++) {
    sprintf(name, "conv[%02d]", i);
    if (i == 7) {
      init_conv_layer(&net.conv[i], name, batch_size, 3, 3, 1, 1, 1, 1, 256, 512, 28, 28);
    }
    else {
      init_conv_layer(&net.conv[i], name, batch_size, 3, 3, 1, 1, 1, 1, 512, 512, 28, 28);
    }

    sprintf(name, "bias[%02d]", i);
    init_bias_layer(&net.bias[i], name, batch_size, 512, 28, 28);

    sprintf(name, "relu[%02d]", i);
    init_act_layer(&net.relu[i], name, batch_size, 512, 28, 28, RELU_T);
  }

  sprintf(name, "pool[3]");
  init_pool_layer(&net.pool[3], name, batch_size, 2, 2, 0, 0, 2, 2, 512, 28, 28, MAX_T);

  for (int i = 10; i < 13; i++) {
    sprintf(name, "conv[%02d]", i);
    init_conv_layer(&net.conv[i], name, batch_size, 3, 3, 1, 1, 1, 1, 512, 512, 14, 14);

    sprintf(name, "bias[%02d]", i);
    init_bias_layer(&net.bias[i], name, batch_size, 512, 14, 14);

    sprintf(name, "relu[%02d]", i);
    init_act_layer(&net.relu[i], name, batch_size, 512, 14, 14, RELU_T);
  }

  sprintf(name, "pool[4]");
  init_pool_layer(&net.pool[4], name, batch_size, 2, 2, 0, 0, 2, 2, 512, 14, 14, MAX_T);

  for (int i = 0; i < 2; i++) {
    sprintf(name, "fc[%d]", i);
    if (i == 0) {
      init_fc_layer(&net.fc[i], name, batch_size, 512 * 7 * 7, 4096);
    }
    else {
      init_fc_layer(&net.fc[i], name, batch_size, 4096, 4096);
    }

    sprintf(name, "fc_bias[%d]", i);
    init_bias_layer(&net.fc_bias[i], name, batch_size, 4096, 1, 1);

    sprintf(name, "fc_relu[%d]", i);
    init_act_layer(&net.fc_relu[i], name, batch_size, 4096, 1, 1, RELU_T);
  }

  sprintf(name, "fc[2]");
  init_fc_layer(&net.fc[2], name, batch_size, 4096, 1000);

  sprintf(name, "fc_bias[2]");
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
  for (int i = 0, j = 0; i < 13; i++) {
    if (i == 0) {
      CONNECT_FROM_INPUT(net.input, net.conv[i]);
    }
    else if (i == 2 || i == 4 || i == 7 || i == 10 || i == 13) {
      CONNECT(net.pool[j], net.conv[i]);
      j++;
    }
    else {
      CONNECT(net.relu[i-1], net.conv[i]);
    }

    CONNECT_WITH_BIAS(net.conv[i], net.bias[i], net.relu[i]);

    if (i == 1 || i == 3 || i == 6 || i == 9 || i == 12) {
      CONNECT(net.relu[i], net.pool[j]);
    }
  }

  CONNECT(net.pool[4], net.fc[0]);

  for (int i = 0; i < 2; i++) {
    CONNECT_WITH_BIAS(net.fc[i], net.fc_bias[i], net.fc_relu[i]);
    CONNECT(net.fc_relu[i], net.fc[i+1]);
  }

  CONNECT_WITH_BIAS(net.fc[2], net.fc_bias[2], net.softmax);
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

  vgg_init(params.batch_size);
  vgg_connect();

  alloc_buffer_by_type(WORK_SPACE);

  int num_batches = num_train_image / params.batch_size;
  fprintf(stderr, "total iteration : %d\n", num_batches);

  size_t sz = vgg_get_param_size();
  float *param_in = (float *)malloc(sz);
  float *param_out = (float *)malloc(sz);
  float *param_result = (float *)malloc(sz);

  INITIALIZE_RAND(param_in, sz / sizeof(float));
  vgg_init_param(param_in);

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

      vgg_copy_input(data_in, label_in);

      if (synch_iteration) {
        clock_gettime(CLOCK_MONOTONIC, &t_end[COPY_INPUT]);
        elapsed_time[COPY_INPUT] += diff_timespec_ms(t_begin[COPY_INPUT], t_end[COPY_INPUT]);
      }

      if (is_first) {
        clock_gettime(CLOCK_MONOTONIC, &t_begin[HEAD_ITERATION]);
      }

      vgg_forward();

#ifdef PRINT_LOSS
      float l = get_loss(&net.softmax, label_in);
      /* MPI */
      if (node_id == 0) {
        printf("loss for %d/%d : %f\n", b, num_batches, l);
      }
#endif

      vgg_backward();

      if (is_first) {
        synch_device();
        clock_gettime(CLOCK_MONOTONIC, &t_end[HEAD_ITERATION]);
        elapsed_time[HEAD_ITERATION] = diff_timespec_ms(t_begin[HEAD_ITERATION], t_end[HEAD_ITERATION]);
#ifdef TIME_LAYER
        vgg_clear_time();
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
    vgg_print_time();
#endif
  }

  vgg_get_param(param_out);

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
