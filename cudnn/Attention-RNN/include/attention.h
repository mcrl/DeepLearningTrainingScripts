#ifndef __ATTENTION_H__
#define __ATTENTION_H__

#include <stdbool.h>
#include "tensor.h"

typedef struct _attention_t attention_t;

attention_t *attention_create(int input_embed_dim, int source_embed_dim, int output_embed_dim, bool bias);
void attention_free(attention_t *attention);
tensor_t* attention_forward(attention_t *attention, tensor_t *x, tensor_t *source_hids, tensor_t *encoder_pading_mask);
tensor_t* attention_inference(attention_t *attention, tensor_t *x, tensor_t *source_hids, tensor_t *encoder_pading_mask);
tensor_pair_t attention_backward(attention_t *attention, tensor_t *dy);
size_t attention_params(attention_t *attention);
void attention_clear(attention_t *attention);
void attention_zerograd(attention_t *attention);
void attention_update(attention_t *attention, int N);
size_t attention_param_init(attention_t *attention, float *param, float *dparam);

#endif //__ATTENTION_H__
