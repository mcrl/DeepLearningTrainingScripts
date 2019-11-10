#ifndef __DROPOUT_H__
#define __DROPOUT_H__

#include "tensor.h"
typedef struct _droptout_t dropout_t;

struct dropout_options;

//
// create dropout object
//
dropout_t* dropout_create(double rate);

//
// destroy dropout object
//
void dropout_free(dropout_t *dropout);

//
// dropout forward for training
//
tensor_t* dropout_forward(dropout_t *droptout, tensor_t *x);

//
// dropout forward for inference
//
tensor_t* dropout_inference(dropout_t *droptout, tensor_t *x);

//
// dropout backward
//
tensor_t* dropout_backward(dropout_t *droptout, tensor_t *dy);

//
// clear temporary resources for the next iteration
//
void dropout_clear(dropout_t *dropout);

//
// set gradients to zero
//
void dropout_zerograd(dropout_t *dropout);

//
// update weights with gradients
//
void dropout_update(dropout_t *dropout, int N);

//
// initialize dropout params. 
// returns (number of params) * 2 for parameter and weight buffer
//
size_t dropout_init_params(dropout_t *dropout, float *param, float *dparam);
//
// get the number of params
//
size_t dropout_params(dropout_t *dropout);


#endif //__DROPOUT_H__

