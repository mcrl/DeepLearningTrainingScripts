#ifndef _CNN_H_
#define _CNN_H_

#include "params.h"

extern float *d;
extern float *p;

void cnn_train(int num_train_image, float *train_data, int *train_label);

#endif // _CNN_H_
