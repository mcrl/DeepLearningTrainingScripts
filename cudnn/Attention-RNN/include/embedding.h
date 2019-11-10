#ifndef __EMBEDDING_H__
#define __EMBEDDING_H__

#include "tensor.h"
typedef struct _embedding_t embedding_t;

//
// create linear (matmul) object
//
embedding_t* embedding_create(int num_embeddings, int embed_dim, int padding_idx);

//
// destroy linear object
//
void embedding_free(embedding_t *embedding);

//
// linear forward for training
//
tensor_t* embedding_forward(embedding_t *embedding, tensor_t *x);

//
// linear forward for inference
//
tensor_t* embedding_inference(embedding_t *embedding, tensor_t *x);

//
// linear backward
//
tensor_t* embedding_backward(embedding_t *embedding, tensor_t *dy);

//
// clear temporary resources for the next iteration
//
void embedding_clear(embedding_t *embedding);

//
// set gradients to zero
//
void embedding_zerograd(embedding_t *embedding);

//
// update weights with gradients
//
void embedding_update(embedding_t *embedding, int N);

//
// initialize embedding params. 
// returns (number of params) * 2 for parameter and weight buffer
//
size_t embedding_param_init(embedding_t *embedding, float *param, float *dparam);

//
// get the number of params
//
size_t embedding_params(embedding_t *embedding);

#endif //__EMBEDDING_H__
