#include <stdlib.h>
#include <assert.h>
#include <cudnn.h>

#include "tstack.h"
#include "optimizer.h"
#include "lstm_cell.h"
#include "tensor.h"
#include "rstack.h"
#include "tstack.h"
#include "snudnn.h"
#include "gpu.h"
#include "utils.h"

struct _lstm_cell_t {
	int input_size;
	int hidden_size;

	void *w;
	void *dw;

	size_t param_size;

	void *workspace;
	size_t workspace_size;

	struct rstack *rstack;

	struct tstack *x_st;
	struct tstack *hx_st;
	struct tstack *cx_st;
	struct tstack *y_st;
	struct tstack *dy_st;
	struct tstack *hy_st;
	struct tstack *cy_st;
	struct tstack *dx_st;
	struct tstack *dhx_st;
	struct tstack *dcx_st;

	cudnnRNNDescriptor_t rnn_desc;

	cudnnFilterDescriptor_t w_desc;
};

static void rnn_init(lstm_cell_t *lstm)
{
	cudnnDropoutDescriptor_t dropout_desc;
	chkCUDNN(cudnnCreateDropoutDescriptor(&dropout_desc));
	chkCUDNN(cudnnSetDropoutDescriptor(dropout_desc, cudnn_handle, 0.0, NULL, 0, 0));
	chkCUDNN(cudnnCreateRNNDescriptor(&lstm->rnn_desc));
	chkCUDNN(cudnnSetRNNDescriptor(cudnn_handle,
				lstm->rnn_desc, 
				lstm->hidden_size,
				1,
				dropout_desc,
				CUDNN_LINEAR_INPUT,
				CUDNN_UNIDIRECTIONAL,
				CUDNN_LSTM,
				CUDNN_RNN_ALGO_STANDARD,
				CUDNN_DATA_FLOAT));

	chkCUDNN(cudnnSetRNNBiasMode(lstm->rnn_desc, CUDNN_RNN_DOUBLE_BIAS));

	lstm->x_st = tstack_create(8);
	lstm->hx_st = tstack_create(8);
	lstm->cx_st = tstack_create(8);
	lstm->y_st = tstack_create(8);
	lstm->dy_st = tstack_create(8);
	lstm->hy_st = tstack_create(8);
	lstm->cy_st = tstack_create(8);
	lstm->dx_st = tstack_create(8);
	lstm->dhx_st = tstack_create(8);
	lstm->dcx_st = tstack_create(8);

}

static void param_init(lstm_cell_t *lstm)
{
	cudnnTensorDescriptor_t x_desc;
	chkCUDNN(cudnnCreateTensorDescriptor(&x_desc));
	int dimA[]    = { 1, lstm->input_size, 1 };
	int strideA[] = { lstm->input_size, 1, 1 };
	chkCUDNN(cudnnSetTensorNdDescriptor(x_desc, CUDNN_DATA_FLOAT, 3, dimA, strideA));


	chkCUDNN(cudnnGetRNNParamsSize(cudnn_handle, lstm->rnn_desc, x_desc, &lstm->param_size, CUDNN_DATA_FLOAT));
	int nelem = lstm->param_size / sizeof(float);
	int filterDimA[] = { 1, 1, nelem };
	chkCUDNN(cudnnCreateFilterDescriptor(&lstm->w_desc));
	chkCUDNN(cudnnSetFilterNdDescriptor(lstm->w_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, 3, filterDimA));
	chkCUDNN(cudnnDestroyTensorDescriptor(x_desc));

}

static void lstm_cell_init(lstm_cell_t *lstm, int input_size, int hidden_size)
{
	memset(lstm, 0, sizeof(lstm_cell_t));

	lstm->input_size = input_size;
	lstm->hidden_size = hidden_size;
	lstm->rstack = rstack_create(8);

	rnn_init(lstm);
	param_init(lstm);
}

static void lstm_cell_release(lstm_cell_t *lstm)
{
}

lstm_cell_t* lstm_cell_create(int input_size, int hidden_size)
{
	lstm_cell_t *lstm = malloc(sizeof(lstm_cell_t));
	lstm_cell_init(lstm, input_size, hidden_size);
	return lstm;
}

void lstm_cell_free(lstm_cell_t *lstm)
{
	lstm_cell_release(lstm);
	free(lstm);
}

static void workspace_prepare(lstm_cell_t *lstm, tensor_t *x)
{
	// must be called after descriptor_prepare
	
	tensor_unsqueeze(x, 0);
	cudnnTensorDescriptor_t x_desc = tensor_descriptor(x);
	tensor_squeeze(x, 0);

	size_t workspace_size;
	chkCUDNN(cudnnGetRNNWorkspaceSize(cudnn_handle, lstm->rnn_desc, 1, &x_desc, &workspace_size));
	lstm->workspace = get_workspace(workspace_size);
	if (lstm->workspace_size < workspace_size)
		lstm->workspace_size = workspace_size;

	size_t reserve_size;
	chkCUDNN(cudnnGetRNNTrainingReserveSize(cudnn_handle, lstm->rnn_desc, 1, &x_desc, &reserve_size));
	rstack_push(lstm->rstack, reserve_size);
}

tensor_pair_t lstm_cell_forward(lstm_cell_t *lstm, tensor_t *x, tensor_t *hx, tensor_t *cx)
{
	// x  : batch_size x input_size
	// hx : batch_size x hidden_size
	// cx : batch_size x hidden_size
	//
	// hy : batch_size x hidden_size
	// cy : batch_size x hidden_size

	assert(x != NULL);

	int batch_size = tensor_size(x, 0);
	int input_size = tensor_size(x, 1);
	int hidden_size = lstm->hidden_size;
	assert(input_size == lstm->input_size);
	//assert(hx != NULL);
	//assert(cx != NULL);

	workspace_prepare(lstm, x);

	int sizes[] = { batch_size, hidden_size };
	tensor_t *y  = tensor_create(sizes, 2);
	tensor_t *hy = tensor_create(sizes, 2);
	tensor_t *cy = tensor_create(sizes, 2);

	const cudnnTensorDescriptor_t x_desc = tensor_descriptor(x);
	const cudnnTensorDescriptor_t y_desc = tensor_descriptor(y);

	tensor_unsqueeze(hy, 0);
	const cudnnTensorDescriptor_t h_desc = tensor_descriptor(hy);
	chkCUDNN(cudnnRNNForwardTraining(cudnn_handle, lstm->rnn_desc,
				1,
				&x_desc, tensor_mem(x),
				h_desc, tensor_mem(hx),
				h_desc, tensor_mem(cx),
				lstm->w_desc, lstm->w,
				&y_desc, tensor_mem(y),
				h_desc, tensor_mem(hy),
				h_desc, tensor_mem(cy),
				lstm->workspace, lstm->workspace_size,
				rstack_top(lstm->rstack), rstack_top_size(lstm->rstack)));
	tensor_squeeze(hy, 0);

	tensor_pair_t lstm_cell_out = {
		.first = hy,
		.second = cy
	};

	tstack_push(lstm->x_st, x);
	tstack_push(lstm->hx_st, hx);
	tstack_push(lstm->cx_st, cx);
	tstack_push(lstm->y_st, y);
	tstack_push(lstm->hy_st, hy);
	tstack_push(lstm->cy_st, cy);

	return lstm_cell_out;
}

tensor_pair_t lstm_cell_inference(lstm_cell_t *lstm, tensor_t *x, tensor_t *hx, tensor_t *cx)
{
	int batch_size = tensor_size(x, 0);
	int input_size = tensor_size(x, 1);
	int hidden_size = lstm->hidden_size;
	assert(input_size == lstm->input_size);
	assert(x != NULL);
	assert(hx != NULL);
	assert(cx != NULL);

	workspace_prepare(lstm, x);

	int sizes[] = { batch_size, hidden_size };
	tensor_t *y = tensor_create(sizes, 2);
	tensor_t *hy = tensor_samesize(hx);
	tensor_t *cy = tensor_samesize(cx);

	const cudnnTensorDescriptor_t x_desc = tensor_descriptor(x);
	const cudnnTensorDescriptor_t y_desc = tensor_descriptor(y);

	tensor_unsqueeze(hy, 0);
	const cudnnTensorDescriptor_t h_desc = tensor_descriptor(hy);
	chkCUDNN(cudnnRNNForwardInference(cudnn_handle, lstm->rnn_desc,
				1,
				&x_desc, tensor_mem(x),
				h_desc, tensor_mem(hx),
				h_desc, tensor_mem(cx),
				lstm->w_desc, lstm->w,
				&y_desc, tensor_mem(y),
				h_desc, tensor_mem(hy),
				h_desc, tensor_mem(cy),
				lstm->workspace, lstm->workspace_size));
	tensor_squeeze(hy, 0);

	tensor_pair_t lstm_cell_out = {
		.first = hy,
		.second = cy
	};

	tstack_push(lstm->x_st, x);
	tstack_push(lstm->hx_st, hx);
	tstack_push(lstm->cx_st, cx);
	tstack_push(lstm->y_st, y);
	tstack_push(lstm->hy_st, hy);
	tstack_push(lstm->cy_st, cy);

	return lstm_cell_out;
}

tensor_triple_t lstm_cell_backward(lstm_cell_t *lstm, tensor_t *dhy, tensor_t *dcy)
{
	tensor_t *dy = tensor_samesize(tstack_top(lstm->y_st));
	tensor_init_constf(dy, 0.0f);

	tensor_t *dx = tensor_samesize(tstack_top(lstm->x_st));
	tensor_t *dhx = tensor_samesize(tstack_top(lstm->hx_st));
	tensor_t *dcx = tensor_samesize(tstack_top(lstm->cx_st));

	cudnnTensorDescriptor_t x_desc = tensor_descriptor(tstack_top(lstm->x_st));
	cudnnTensorDescriptor_t y_desc = tensor_descriptor(tstack_top(lstm->y_st));

	tensor_unsqueeze(dhy, 0);
	cudnnTensorDescriptor_t h_desc = tensor_descriptor(dhy);
	chkCUDNN(cudnnRNNBackwardData(cudnn_handle,
			lstm->rnn_desc, 1,
			&y_desc, tstack_topmem(lstm->y_st),
			&y_desc, tensor_mem(dy),
			h_desc, tensor_mem(dhy),
			h_desc, tensor_mem(dcy),
			lstm->w_desc, lstm->w,
			h_desc, tstack_topmem(lstm->hx_st),
			h_desc, tstack_topmem(lstm->cx_st),
			&x_desc, tensor_mem(dx),
			h_desc, tensor_mem(dhx),
			h_desc, tensor_mem(dcx),
			lstm->workspace, lstm->workspace_size,
			rstack_top(lstm->rstack), rstack_top_size(lstm->rstack)));

	chkCUDNN(cudnnRNNBackwardWeights(cudnn_handle,
			lstm->rnn_desc, 1, 
			&x_desc, tstack_topmem(lstm->x_st),
			h_desc, tstack_topmem(lstm->hx_st),
			&y_desc, tstack_topmem(lstm->y_st),
			lstm->workspace, lstm->workspace_size,
			lstm->w_desc, lstm->dw,
			rstack_top(lstm->rstack), rstack_top_size(lstm->rstack)));
	tensor_squeeze(dhy, 0);


	rstack_pop(lstm->rstack);

	tstack_pop(lstm->x_st);
	tstack_pop(lstm->hx_st);
	tstack_pop(lstm->cx_st);

	tstack_pop(lstm->y_st);
	tstack_pop(lstm->hy_st);
	tstack_pop(lstm->cy_st);


	tstack_push(lstm->dy_st, dy);
	tstack_push(lstm->dx_st, dx);
	tstack_push(lstm->dhx_st, dhx);
	tstack_push(lstm->dcx_st, dcx);

	tensor_triple_t triple = {
		.first = dx,
		.second = dhx,
		.third = dcx,
	};
	return triple;
}

size_t lstm_cell_params(lstm_cell_t *lstm_cell)
{
	return lstm_cell->param_size / sizeof(float);
}

size_t lstm_cell_param_init(lstm_cell_t *lstm_cell, float *param, float *dparam)
{
	lstm_cell->w = param;
	lstm_cell->dw = dparam;
	int N = lstm_cell->param_size / sizeof(float);
	snudnn_uniform(lstm_cell->w, N, -0.1, 0.1);
	return N;
}

void lstm_cell_clear(lstm_cell_t *lstm)
{
	tstack_clear_s(lstm->x_st);
	tstack_clear_s(lstm->hx_st);
	tstack_clear_s(lstm->cx_st);

	tstack_clear(lstm->y_st);
	tstack_clear(lstm->dy_st);
	tstack_clear(lstm->hy_st);
	tstack_clear(lstm->cy_st);

	tstack_clear(lstm->dx_st);
	tstack_clear(lstm->dhx_st);
	tstack_clear(lstm->dcx_st);

	rstack_clear(lstm->rstack);
}

void lstm_cell_zerograd(lstm_cell_t *lstm_cell)
{
	chkCUDA(cudaMemset(lstm_cell->dw, 0, lstm_cell->param_size));
}


void lstm_cell_update(lstm_cell_t *lstm_cell, int N)
{
	size_t nelem = lstm_cell->param_size / sizeof(float);
	optimizer_step(lstm_cell->dw, lstm_cell->w, nelem, N);
}
