#ifndef __ENCODER_H__
#define __ENCODER_H__

#include "tensor.h"
typedef struct _encoder_t encoder_t;

struct encoder_options;

//
// create encoder object
//
encoder_t* encoder_create(struct encoder_options *options);

//
// destroy encoder object
//
void encoder_free(encoder_t *encoder);

//
// encoder forward for training
//
tensor_quadruple_t encoder_forward(encoder_t *encoder,
		tensor_t *x,
		const int *seq_len_array);

//
// encoder forward for inference
//
tensor_quadruple_t encoder_inference(encoder_t *encoder,
		tensor_t *x,
		const int *seq_len_array);

//
// encoder backward
//
tensor_t* encoder_backward(encoder_t *encoder, tensor_t *dy, tensor_t *dhy, tensor_t *dcy);

//
// clear temporary resources for the next iteration
//
void encoder_clear(encoder_t *encoder);

//
// set gradients to zero
//
void encoder_zerograd(encoder_t *encoder);

//
// update weights with gradients
//
void encoder_update(encoder_t *encoder, int N);

//
// initialize encoder params. 
// returns (number of params) * 2 for parameter and weight buffer
//
size_t encoder_init_params(encoder_t *encoder, float *param, float *dparam);

//
// get the number of params
//
size_t encoder_params(encoder_t *encoder);
#endif //__ENCODER_H__

