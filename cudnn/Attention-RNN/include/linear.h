#ifndef __LINEAR_H__
#define __LINEAR_H__

#include <stdbool.h>
#include "tensor.h"

typedef struct _linear_t linear_t;

//
// create linear (matmul) object
//
linear_t* linear_create(int input_dim, int output_dim, bool bias);

//
// destroy linear object
//
void linear_free(linear_t *linear);

//
// linear forward for training
//
tensor_t* linear_forward(linear_t *linear, tensor_t *x);

//
// linear forward for inference
//
tensor_t* linear_inference(linear_t *linear, tensor_t *x);

//
// linear backward
//
tensor_t* linear_backward(linear_t *linear, tensor_t *dy);

//
// clear temporary resources for the next iteration
//
void linear_clear(linear_t *linear);

//
// set gradients to zero
//
void linear_zerograd(linear_t *linear);

//
// update weights with gradients
//
void linear_update(linear_t *linear, int N);

//
// initialize linear params. 
// returns (number of params) * 2 for parameter and weight buffer
//
size_t linear_init_params(linear_t *linear, float *param, float *dparam);

//
// get the number of params
//
size_t linear_params(linear_t *linear);

#endif //__LINEAR_H__
