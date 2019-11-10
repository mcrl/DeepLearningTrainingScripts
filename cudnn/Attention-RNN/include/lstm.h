#ifndef __LSTM_H__
#define __LSTM_H__

#include <stdbool.h>
#include "tensor.h"

typedef struct _lstm_t lstm_t;

lstm_t* lstm_create(int input_size, int hidden_size, int num_layers, double rate);
void lstm_free(lstm_t *lstm);
tensor_triple_t lstm_forward(lstm_t *lstm, tensor_t *x, tensor_t *hx, tensor_t *cx, const int *seq_len_arr);
tensor_triple_t lstm_inference(lstm_t *lstm, tensor_t *x, tensor_t *hx, tensor_t *cx, const int *seq_len_arr);
tensor_triple_t lstm_backward(lstm_t *lstm, tensor_t *dy, tensor_t *dhy, tensor_t *dcy);
size_t lstm_params(lstm_t *lstm);
void lstm_clear(lstm_t *lstm);
void lstm_zerograd(lstm_t *lstm);
void lstm_update(lstm_t *lstm, int N);
size_t lstm_param_init(lstm_t *lstm, float *param, float *dparam);

#endif //__LSTM_H__
