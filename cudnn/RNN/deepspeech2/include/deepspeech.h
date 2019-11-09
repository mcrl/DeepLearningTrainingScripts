#ifndef _DEEPSPEECH_H_
#define _DEEPSPEECH_H_

#include "params.h"
#include "dataset.h"

extern float *d;
extern float *p;

void deepspeech_train(dataset_t *dataset_train, int world_rank,
  int world_size, int num_dev_per_node);

void deepspeech_eval(dataset_t *dataset_val, int world_rank,
  int world_size, int num_dev_per_node);

#endif

