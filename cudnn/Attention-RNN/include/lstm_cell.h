#ifndef __LSTM_CELL_H__
#define __LSTM_CELL_H__

#include "tensor.h"
typedef struct _lstm_cell_t lstm_cell_t;


lstm_cell_t* lstm_cell_create(int input_size, int hidden_size);
void lstm_cell_free(lstm_cell_t *lstm_cell);
tensor_pair_t lstm_cell_forward(lstm_cell_t *lstm, tensor_t *x, tensor_t *hx, tensor_t *cx);
tensor_pair_t lstm_cell_inference(lstm_cell_t *lstm, tensor_t *x, tensor_t *hx, tensor_t *cx);
tensor_triple_t lstm_cell_backward(lstm_cell_t *lstm, tensor_t *dhy, tensor_t *dcy);
size_t lstm_cell_params(lstm_cell_t *lstm_cell);
void lstm_cell_clear(lstm_cell_t *lstm_cell);
void lstm_cell_zerograd(lstm_cell_t *lstm_cell);
void lstm_cell_update(lstm_cell_t *lstm_cell, int N);
size_t lstm_cell_param_init(lstm_cell_t *lstm_cell, float *param, float *dparam);
#endif //__LSTM_CELL_H__
