#ifndef __DECODER_H__
#define __DECODER_H__

#include "tensor.h"

typedef struct _decoder_t decoder_t;

struct decoder_options;

//
// create decoder object
//
decoder_t* decoder_create(struct decoder_options opts);

//
// destroy decoder object
//
void decoder_free(decoder_t *decoder);

//
// decoder forward for training
//
tensor_t* decoder_forward(decoder_t *decoder,
		tensor_t *prev_output_tokens,
		tensor_t *encoder_outs,
		tensor_t *encoder_hiddens,
		tensor_t *encoder_cells,
		tensor_t *encoder_padding_mask);


//
// decoder forward for inference
//
tensor_t* decoder_inference(decoder_t *decoder,
		tensor_t *prev_output_tokens,
		tensor_t *encoder_outs,
		tensor_t *encoder_hiddens,
		tensor_t *encoder_cells,
		tensor_t *encoder_padding_mask);

//
// decoder backward
//
tensor_triple_t decoder_backward(decoder_t *decoder, tensor_t *dy);

//
// clear temporary resources for the next iteration
//
void decoder_clear(decoder_t *decoder);

//
// set gradients to zero
//
void decoder_zerograd(decoder_t *decoder);

//
// update weights with gradients
//
void decoder_update(decoder_t *decoder, int N);

//
// initialize decoder parameters
// returns the number of params
//
size_t decoder_init_params(decoder_t *decoder, float *param, float *dparam);

//
// get the number of params
//
size_t decoder_params(decoder_t *decoder);
#endif //__DECODER_H__

