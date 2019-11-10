#ifndef __LSTM_ATTN_T__
#define __LSTM_ATTN_T__

#include "tensor.h"

typedef struct _lstm_attn_t lstm_attn_t;


lstm_attn_t* lstm_attn_create(int input_size, int hidden_size, int num_layers,
		double rate, bool bidirectional, int max_len, int padding_idx);
void lstm_attn_free(lstm_attn_t *lstm_attn);
tensor_t* lstm_attn_forward(lstm_attn_t *lstm_attn,
		tensor_t *input,
		tensor_t *encoder_outs,
		tensor_t *encoder_hiddens,
		tensor_t *encoder_cells,
		tensor_t *encoder_padding_mask);
tensor_t* lstm_attn_inference(lstm_attn_t *lstm_attn,
		tensor_t *input,
		tensor_t *encoder_outs,
		tensor_t *encoder_hiddens,
		tensor_t *encoder_cells,
		tensor_t *encoder_padding_mask);
tensor_quadruple_t lstm_attn_backward(lstm_attn_t *lstm_attn, tensor_t *dy);
size_t lstm_attn_params(lstm_attn_t *lstm_attn);
void lstm_attn_clear(lstm_attn_t *lstm_attn);
void lstm_attn_zerograd(lstm_attn_t *lstm_attn);
void lstm_attn_update(lstm_attn_t *lstm_attn, int N);
size_t lstm_attn_param_init(lstm_attn_t *lstm_attn, float *param, float *dparam);

#endif //__LSTM_ATTN_T__
