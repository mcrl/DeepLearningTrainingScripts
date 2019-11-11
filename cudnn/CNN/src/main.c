#include <stdio.h>
#include <assert.h>
#include <stdlib.h>

#include "cnn.h"
#include "layer.h"
#include "params.h"

int num_train_image;
float *train_image;
int *train_label;

struct params params = {
  .seed = 0xdeadbeef,
  .epochs = 100,
  .width = 224,
  .height = 224,
  .channel = 3,
  .learning_anneal = 1.001,
  .learning_rate = 0.001,
  .momentum = 0.9,
  .weight_decay = 1e-5,
  .max_norm = 400,
  .l2 = 0,
  .batch_size = 4,
  .num_batch_per_epoch = 100,
  .augment = true,
  .exit_at_acc = true,
  .result = NULL,
  .input_param = NULL,
  .result_output = NULL
};

void load_input(
    const char *file_image, const char *file_label, float **image, int **label)
{
  FILE *fimage = fopen(file_image, "rb");
  FILE *flabel = fopen(file_label, "rb");

  if (!fimage) {
    fprintf(stderr, "%s does not exist\n", file_image);
    exit(0);
  }

  if (!flabel) {
    fprintf(stderr, "%s does not exist\n", file_label);
    exit(0);
  }

  fprintf(stderr, "num_image : %d\n", num_train_image);

  size_t s1, s2, s3, s4;
  s1 = sizeof(float) * params.width * params.height * params.channel * num_train_image;
  s2 = sizeof(int) * num_train_image;

  *image = (float *)malloc(s1);
  *label = (int *)malloc(s2);

  s3 = fread(*image, 1, s1, fimage);
  s4 = fread(*label, 1, s2, flabel);

  if (s1 != s3) {
    fprintf(stderr, "%s is too small\n", file_image);
    exit(0);
  }

  if (s2 != s4) {
    fprintf(stderr, "%s is too small\n", file_label);
    exit(0);
  }

  fclose(fimage);
  fclose(flabel);
}

int main(int argc, char *argv[])
{
  if (argc < 6) {
    fprintf(stderr, "%s [batch_size] [iteration] [image] [label] [result]\n", argv[0]);
    return 0;
  }

  params_modify();
  params.batch_size = atoi(argv[1]);
  params.num_batch_per_epoch = atoi(argv[2]);
  #ifdef PRINT_LOSS
  params.epochs = 100;
  #else
  params.epochs = 10;
  #endif
  params.result = argv[5];

  num_train_image = params.batch_size * params.num_batch_per_epoch;

  load_input(argv[3], argv[4], &train_image, &train_label);

  cnn_train(num_train_image, train_image, train_label);

  return 0;
}
