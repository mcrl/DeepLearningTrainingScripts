#ifndef _PARAMS_H_
#define _PARAMS_H_

#include <stdbool.h>

struct params {
  unsigned long long seed;
  unsigned long long epochs;
  int width;
  int height;
  int channel;
  float learning_anneal;
  float learning_rate;
  float momentum;
  float weight_decay;
  unsigned long long max_norm;
  unsigned long long l2; 
  unsigned long long batch_size;
  unsigned long long num_batch_per_epoch;
  bool augment;
  bool exit_at_acc;
  char *result;
  char *input_param;
  char *result_output;
};

extern struct params params;

#endif // _PARAMS_H_
