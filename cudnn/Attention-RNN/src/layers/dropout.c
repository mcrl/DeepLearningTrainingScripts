#include "dropout.h"
#include "gpu.h"
#include "snudnn.h"

#include "rstack.h"
#include "tstack.h"

#include <stdlib.h>

struct _droptout_t {
	float rate;
	unsigned long long seed;

	void *state;
	size_t state_size;

	struct rstack *rstack;

	tensor_t *y;
	tensor_t *dx;

	struct tstack *tstack;

	cudnnDropoutDescriptor_t desc;
};

static void dropout_init(dropout_t *dropout, double rate)
{
	// TODO: choose appropriate random seed
	dropout->seed = 0;

	dropout->rate = rate;
	chkCUDNN(cudnnCreateDropoutDescriptor(&dropout->desc));
	chkCUDNN(cudnnDropoutGetStatesSize(cudnn_handle, &dropout->state_size));
	chkCUDA(cudaMalloc(&dropout->state, dropout->state_size));
	chkCUDNN(cudnnSetDropoutDescriptor(dropout->desc, cudnn_handle, rate,
				dropout->state, dropout->state_size, dropout->seed));

	dropout->rstack = rstack_create(8);
	dropout->tstack = tstack_create(8);
}

static void dropout_release(dropout_t *dropout)
{
	chkCUDNN(cudnnDestroyDropoutDescriptor(dropout->desc));
	chkCUDA(cudaFree(dropout->state));
}

dropout_t* dropout_create(double rate)
{
	dropout_t *dropout = malloc(sizeof(dropout_t));
	dropout_init(dropout, rate);
	return dropout;
}

void dropout_free(dropout_t *dropout)
{
	dropout_release(dropout);
	free(dropout);
}

static void prepare_reserve(dropout_t *dropout, tensor_t *x)
{
    size_t reserve_size;
    chkCUDNN(cudnnDropoutGetReserveSpaceSize(tensor_descriptor(x), &reserve_size));
    rstack_push(dropout->rstack, reserve_size);
}

tensor_t* dropout_forward(dropout_t *dropout, tensor_t *x)
{
	// (in)
	// 		x : batch_size x data_size
	// (out)
	// 		y : batch_size x data_size
	if (dropout->rate == 0.0f) {
		dropout->y = x;
		return x;
	}

	prepare_reserve(dropout, x);

	tensor_t *y = tensor_samesize(x);
	chkCUDNN(cudnnDropoutForward(cudnn_handle, dropout->desc, 
				tensor_descriptor(x), tensor_mem(x), 
				tensor_descriptor(y), tensor_mem(y),
				rstack_top(dropout->rstack),
				rstack_top_size(dropout->rstack)));
	dropout->y = y;
	tstack_push(dropout->tstack, y);

	return y;
}

tensor_t* dropout_inference(dropout_t *dropout, tensor_t *x)
{
	// Do not apply dropout during inference 
	dropout->y = x;
	return x;
}

tensor_t* dropout_backward(dropout_t *dropout, tensor_t *dy)
{
	// (in)
	// 		dy : batch_size x data_size
	// (out)
	// 		dx : batch_size x data_size
	if (dropout->rate == 0.0f) {
		dropout->dx = dy;
		return dy;
	}

	tensor_t *dx = tensor_samesize(dy);
	chkCUDNN(cudnnDropoutBackward(cudnn_handle,
				dropout->desc,
				tensor_descriptor(dy), tensor_mem(dy),
				tensor_descriptor(dx), tensor_mem(dx),
				rstack_top(dropout->rstack),
				rstack_top_size(dropout->rstack)));
	rstack_pop(dropout->rstack);
	tstack_push(dropout->tstack, dx);
	dropout->dx = dx;
	return dx;
}

size_t dropout_params(dropout_t *dropout)
{
	return 0;
}

size_t dropout_init_params(dropout_t *dropout, float *param, float *dparam)
{
	return 0;
}

void dropout_clear(dropout_t *dropout)
{
	if (dropout->rate != 0.0f)  {
		//tensor_free(dropout->y);
		//tensor_free(dropout->dx);
	}
	tstack_clear(dropout->tstack);
	rstack_clear(dropout->rstack);
}


void dropout_zerograd(dropout_t *dropout)
{
}

void dropout_update(dropout_t *dropout, int N)
{
}
