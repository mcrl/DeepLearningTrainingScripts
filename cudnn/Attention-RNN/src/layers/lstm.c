#include <stdlib.h>
#include <assert.h>
#include <cudnn.h>

#include "optimizer.h"
#include "lstm.h"
#include "dropout.h"
#include "gpu.h"
#include "rstack.h"
#include "snudnn.h"
#include "utils.h"

struct _lstm_t {
	int input_size;
	int hidden_size;
	int num_layers;
	bool has_dropout;
	double dropout;
	float padding_value;

	int prev_seqlen;

	cudnnRNNDescriptor_t static_desc;
	cudnnRNNDescriptor_t *descs;
	cudnnPersistentRNNPlan_t *plans;

	tensor_t *x;
	tensor_t *hx;
	tensor_t *cx;
	tensor_t *y;
	tensor_t *hy;
	tensor_t *cy;
	void *w;

	tensor_t *dx;
	tensor_t *dhx;
	tensor_t *dcx;
	void *dw;

	size_t param_size;

	void *workspace;
	size_t workspace_size;

	struct rstack *rstack;

	cudnnDropoutDescriptor_t dropout_desc;
	void *dropout_state;
	size_t dropout_state_size;

	cudnnRNNDescriptor_t rnn_desc;

	cudnnRNNDataDescriptor_t x_desc;
	cudnnRNNDataDescriptor_t y_desc;

	cudnnFilterDescriptor_t w_desc;
	cudnnFilterDescriptor_t dwdesc;
};

static void dropout_init(lstm_t *lstm)
{
	chkCUDNN(cudnnCreateDropoutDescriptor(&lstm->dropout_desc));
	if (lstm->has_dropout) {
		chkCUDNN(cudnnDropoutGetStatesSize(cudnn_handle, &lstm->dropout_state_size));
		chkCUDA(cudaMalloc(&lstm->dropout_state, lstm->dropout_state_size));
		chkCUDNN(cudnnSetDropoutDescriptor(lstm->dropout_desc, cudnn_handle, lstm->dropout,
					lstm->dropout_state, lstm->dropout_state_size, 0));
	}
}

static void rnn_init(lstm_t *lstm)
{
	lstm->descs = malloc(sizeof(cudnnRNNDescriptor_t) * 128);
	lstm->plans = malloc(sizeof(cudnnPersistentRNNPlan_t) * 128);

	chkCUDNN(cudnnCreateRNNDataDescriptor(&lstm->y_desc));

	chkCUDNN(cudnnCreateRNNDescriptor(&lstm->static_desc));
	chkCUDNN(cudnnSetRNNDescriptor(cudnn_handle,
				lstm->static_desc, 
				lstm->hidden_size,
				lstm->num_layers,
				lstm->dropout_desc,
				CUDNN_LINEAR_INPUT,
				CUDNN_UNIDIRECTIONAL,
				CUDNN_LSTM,
				CUDNN_RNN_ALGO_STANDARD,
				CUDNN_DATA_FLOAT));

	chkCUDNN(cudnnSetRNNBiasMode(lstm->static_desc, CUDNN_RNN_DOUBLE_BIAS));
	chkCUDNN(cudnnSetRNNPaddingMode(lstm->static_desc, CUDNN_RNN_PADDED_IO_ENABLED));
	lstm->rnn_desc = lstm->static_desc;

	chkCUDNN(cudnnCreateRNNDataDescriptor(&lstm->x_desc));
	chkCUDNN(cudnnCreateRNNDataDescriptor(&lstm->y_desc));

}

static void param_init(lstm_t *lstm)
{
	cudnnTensorDescriptor_t x_desc;
	chkCUDNN(cudnnCreateTensorDescriptor(&x_desc));
	int dimA[]    = { 1, lstm->input_size, 1 };
	int strideA[] = { lstm->input_size, 1, 1 };
	chkCUDNN(cudnnSetTensorNdDescriptor(x_desc, CUDNN_DATA_FLOAT, 3, dimA, strideA));


	chkCUDNN(cudnnGetRNNParamsSize(cudnn_handle, lstm->static_desc, x_desc, &lstm->param_size, CUDNN_DATA_FLOAT));
	int nelem = lstm->param_size / sizeof(float);
	int filterDimA[] = { 1, 1, nelem };
	chkCUDNN(cudnnCreateFilterDescriptor(&lstm->w_desc));
	chkCUDNN(cudnnSetFilterNdDescriptor(lstm->w_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, 3, filterDimA));
	chkCUDNN(cudnnDestroyTensorDescriptor(x_desc));
}

static void lstm_init(lstm_t *lstm, int input_size, int hidden_size, int num_layers, double dropout)
{
	memset(lstm, 0, sizeof(lstm_t));

	lstm->input_size = input_size;
	lstm->hidden_size = hidden_size;
	lstm->num_layers = num_layers;
	lstm->has_dropout = num_layers > 1;
	lstm->dropout = dropout;
	lstm->padding_value = 0.0f;

	lstm->rstack = rstack_create(8);
	lstm->prev_seqlen = 0;

	dropout_init(lstm);
	rnn_init(lstm);
	param_init(lstm);
}

static void lstm_release(lstm_t *lstm)
{
}

lstm_t* lstm_create(int input_size, int hidden_size, int num_layers, double dropout)
{
	lstm_t *lstm = malloc(sizeof(lstm_t));
	lstm_init(lstm, input_size, hidden_size, num_layers, dropout);
	return lstm;
}

void lstm_free(lstm_t *lstm)
{
	lstm_release(lstm);
	free(lstm);
}

static void workspace_prepare(lstm_t *lstm, tensor_t *x)
{
	// must be called after descriptor_prepare
	int seq_len = tensor_size(x, 0);
	int batch_size = tensor_size(x, 1);
	int input_size = tensor_size(x, 2);

	cudnnTensorDescriptor_t xdesc[seq_len];

	int dimA[] = { batch_size, input_size, 1 };
	int strideA[] = { input_size, 1, 1 };
	for (int i = 0; i < seq_len; i++) {
		chkCUDNN(cudnnCreateTensorDescriptor(&xdesc[i]));
		chkCUDNN(cudnnSetTensorNdDescriptor(xdesc[i], CUDNN_DATA_FLOAT, 3, dimA, strideA));
	}


	size_t workspace_size;
	chkCUDNN(cudnnGetRNNWorkspaceSize(cudnn_handle, lstm->rnn_desc, seq_len, xdesc,
				&workspace_size));
	lstm->workspace = get_workspace(workspace_size);
	if (lstm->workspace_size < workspace_size)
		lstm->workspace_size = workspace_size;

	size_t reserve_size;
	chkCUDNN(cudnnGetRNNTrainingReserveSize(cudnn_handle, lstm->rnn_desc, seq_len, xdesc, &reserve_size));
	rstack_push(lstm->rstack, reserve_size);

	for (int i = 0; i < seq_len; i++) {
		chkCUDNN(cudnnDestroyTensorDescriptor(xdesc[i]));
	}
}


static void descriptor_prepare(lstm_t *lstm, tensor_t *x, const int *seq_len_arr)
{
	int seq_len = tensor_size(x, 0);
	int batch_size = tensor_size(x, 1);
	int input_size = lstm->input_size;
	int hidden_size = lstm->hidden_size;

	chkCUDNN(cudnnSetRNNDataDescriptor(lstm->x_desc, 
			CUDNN_DATA_FLOAT,
			CUDNN_RNN_DATA_LAYOUT_SEQ_MAJOR_UNPACKED,
			seq_len,
			batch_size,
			input_size,
			seq_len_arr,
			NULL));

	chkCUDNN(cudnnSetRNNDataDescriptor(lstm->y_desc, 
			CUDNN_DATA_FLOAT,
			CUDNN_RNN_DATA_LAYOUT_SEQ_MAJOR_UNPACKED,
			seq_len,
			batch_size,
			hidden_size,
			seq_len_arr,
			&lstm->padding_value));
}

static void lstm_prepare(lstm_t *lstm, tensor_t *x, const int *seq_len_arr)
{
	descriptor_prepare(lstm, x, seq_len_arr);
	workspace_prepare(lstm, x);
}

tensor_triple_t lstm_forward(lstm_t *lstm, tensor_t *x, tensor_t *hx, tensor_t *cx, const int *seq_len_arr)
{
	// x  : seq_len x batch_size x input_size
	// hx : [2*]num_layers x batch_size x hidden_size
	// cx : [2*]num_layers x batch_size x hidden_size
	// seqlen_array: 'batch_size' array of sizes
	// y : seq_len x batch_size x hidden_size

	//int num_layers = lstm->num_layers;
	int seq_len = tensor_size(x, 0);
	int batch_size = tensor_size(x, 1);
	int hidden_size = lstm->hidden_size;
	int num_layers = lstm->num_layers;


	assert(tensor_size(x, 2) == lstm->input_size);
	assert(seq_len > 0);
	assert(batch_size > 0);

	tensor_t *y;
	tensor_t *hy;
	tensor_t *cy;

	lstm_prepare(lstm, x, seq_len_arr);

	int sizes[] = { seq_len, batch_size, hidden_size };
	int hsizes[] = { num_layers, batch_size, hidden_size };
	y = tensor_create(sizes, 3);
	hy = tensor_create(hsizes, 3);
	cy = tensor_create(hsizes, 3);

	chkCUDNN(cudnnRNNForwardTrainingEx(cudnn_handle,
				lstm->rnn_desc,
				lstm->x_desc, tensor_mem(x),
				tensor_descriptor(hy), tensor_mem(hx),
				tensor_descriptor(cy), tensor_mem(cx),
				lstm->w_desc, lstm->w,
				lstm->y_desc, tensor_mem(y),
				tensor_descriptor(hy), tensor_mem(hy),
				tensor_descriptor(cy), tensor_mem(cy),
				NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL,
				lstm->workspace, lstm->workspace_size,
				rstack_top(lstm->rstack), rstack_top_size(lstm->rstack)));


	tensor_triple_t lstm_out = {
		.first = y,
		.second = hy,
		.third = cy
	};

	lstm->x = x;
	lstm->hx = hx;
	lstm->cx = cx;
	lstm->y = y;
	lstm->hy = hy;
	lstm->cy = cy;
	return lstm_out;
}

tensor_triple_t lstm_inference(lstm_t *lstm, tensor_t *x, tensor_t *hx, tensor_t *cx, const int *seq_len_arr)
{
	int seq_len = tensor_size(x, 0);
	int batch_size = tensor_size(x, 1);
	int hidden_size = lstm->hidden_size;
	int num_layers = lstm->num_layers;

	assert(tensor_size(x, 2) == lstm->input_size);
	assert(seq_len > 0);
	assert(batch_size > 0);

	lstm_prepare(lstm, x, seq_len_arr);

	int sizes[] = { seq_len, batch_size, hidden_size };
	int hsizes[] = { num_layers, batch_size, hidden_size };
	tensor_t *y = tensor_create(sizes, 3);
	tensor_t *hy = tensor_create(hsizes, 3);
	tensor_t *cy = tensor_create(hsizes, 3);

	chkCUDNN(cudnnRNNForwardInferenceEx(cudnn_handle,
				lstm->rnn_desc,
				lstm->x_desc, tensor_mem(x),
				tensor_descriptor(hy), tensor_mem(hx),
				tensor_descriptor(cy), tensor_mem(cx),
				lstm->w_desc, lstm->w,
				lstm->y_desc, tensor_mem(y),
				tensor_descriptor(hy), tensor_mem(hy),
				tensor_descriptor(cy), tensor_mem(cy),
				NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL,
				lstm->workspace, lstm->workspace_size));

	tensor_triple_t lstm_out = {
		.first = y,
		.second = hy,
		.third = cy
	};

	lstm->x = x;
	lstm->hx = hx;
	lstm->cx = cx;
	lstm->y = y;
	lstm->hy = hy;
	lstm->cy = cy;
	return lstm_out;
}

tensor_triple_t lstm_backward(lstm_t *lstm, tensor_t *dy, tensor_t *dhy, tensor_t *dcy)
{
	// dy : seq_len x batch_size x hidden_size
	assert(dy != NULL);
	assert(dhy != NULL);

	tensor_t *dx = tensor_samesize(lstm->x);
	tensor_t *dhx = tensor_samesize(dhy);
	tensor_t *dcx = tensor_samesize(dcy);

	chkCUDNN(cudnnRNNBackwardDataEx(cudnn_handle,
			lstm->rnn_desc,
			lstm->y_desc, tensor_mem(lstm->y),
			lstm->y_desc, tensor_mem(dy),
			NULL, NULL,
			tensor_descriptor(lstm->hy), tensor_mem(dhy),
			tensor_descriptor(lstm->cy), tensor_mem(dcy),
			lstm->w_desc, lstm->w,
			tensor_descriptor(lstm->hy), tensor_mem(lstm->hx),
			tensor_descriptor(lstm->cy), tensor_mem(lstm->cx),
			lstm->x_desc, tensor_mem(dx),
			tensor_descriptor(lstm->hy), tensor_mem(dhx),
			tensor_descriptor(lstm->cy), tensor_mem(dcx),
			NULL, NULL,
			lstm->workspace, lstm->workspace_size,
			rstack_top(lstm->rstack), rstack_top_size(lstm->rstack)));

	chkCUDNN(cudnnRNNBackwardWeightsEx(cudnn_handle,
			lstm->rnn_desc,
			lstm->x_desc, tensor_mem(lstm->x),
			tensor_descriptor(lstm->hx), tensor_mem(lstm->hx),
			lstm->y_desc, tensor_mem(lstm->y),
			lstm->workspace, lstm->workspace_size,
			lstm->w_desc, lstm->dw,
			rstack_top(lstm->rstack), rstack_top_size(lstm->rstack)));

	rstack_pop(lstm->rstack);

	lstm->dx = dx;
	lstm->dhx = dhx;
	lstm->dcx = dcx;

	tensor_triple_t triple = {
		.first = dx,
		.second = dhx,
		.third = dcx,
	};
	return triple;
}

size_t lstm_params(lstm_t *lstm)
{
	return lstm->param_size / sizeof(float);
}

size_t lstm_param_init(lstm_t *lstm, float *param, float *dparam)
{
	lstm->w = param;
	lstm->dw = dparam;
	int N = lstm->param_size / sizeof(float);
	snudnn_uniform(lstm->w, N, -0.1, 0.1);
	return N;
}

void lstm_clear(lstm_t *lstm)
{
	tensor_free(lstm->y);
	tensor_free(lstm->hy);
	tensor_free(lstm->cy);

	tensor_free(lstm->dx);
	tensor_free(lstm->dhx);
	tensor_free(lstm->dcx);

	rstack_clear(lstm->rstack);
}

void lstm_zerograd(lstm_t *lstm)
{
	chkCUDA(cudaMemset(lstm->dw, 0, lstm->param_size));
}

void lstm_update(lstm_t *lstm, int N)
{
	size_t nelem = lstm->param_size/sizeof(float);
	optimizer_step(lstm->dw, lstm->w, nelem, N);
}
