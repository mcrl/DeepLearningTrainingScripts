#include <assert.h>
#include <stdlib.h>
#include "linear.h"
#include "gpu.h"
#include "snudnn.h"
#include "utils.h"
#include "optimizer.h"
#include "tstack.h"
#include "cache_allocator.h"

void cuda_bias_forward(const float *bias, float *y, int N, int M);
struct _linear_t {
	int input_dim;
	int output_dim;
	bool has_bias;

	tensor_t *w;
	tensor_t *b;

	tensor_t *dw;
	tensor_t *db;


	tensor_t *x;

	// free list
	tensor_t *y;
	tensor_t *dx;


	struct tstack *tstack;

	cudnnReduceTensorDescriptor_t desc;
};

void linear_init(linear_t *linear, int input_dim, int output_dim, bool bias)
{
	linear->input_dim = input_dim;
	linear->output_dim = output_dim;
	linear->has_bias = bias;


	int sizes[2] = { output_dim, input_dim };

	linear->w = tensor_create_empty(sizes, 2);
	linear->dw = tensor_create_empty(sizes, 2);
	if (linear->has_bias) {
		int sizes[] = { 1, output_dim };
		linear->b = tensor_create_empty(sizes, 2);
		linear->db = tensor_create_empty(sizes, 2);
	}

	linear->tstack = tstack_create(8);
}

void linear_release(linear_t *linear)
{
	tensor_free(linear->w);
	if (linear->has_bias)
		tensor_free(linear->b);
}

linear_t* linear_create(int input_dim, int output_dim, bool bias)
{
	linear_t *linear = malloc(sizeof(linear_t));
	linear_init(linear, input_dim, output_dim, bias);
	return linear;
}

void linear_free(linear_t *linear)
{
	linear_release(linear);
	free(linear);
}

void linear_clean(linear_t *linear)
{
	tensor_free(linear->y);
	tensor_free(linear->dx);
}

tensor_t* linear_forward(linear_t *linear, tensor_t *x)
{
	// x : batch_size x input_dim
	// y : batch_size x output_dim

	int output_dim = linear->output_dim;
	int input_dim = linear->input_dim;
	int dim = tensor_dim(x);

	int effective_batch = tensor_nelem(x) / input_dim;
	assert(input_dim == tensor_size(x, dim-1));

	int sizes[dim];
	for (int i = 0; i < dim-1; i++) {
		sizes[i] = tensor_size(x, i);
	}
	sizes[dim-1] = output_dim;

	tensor_t *y = tensor_create(sizes, dim);
	tstack_push(linear->tstack, y);

	chkCUBLAS(cublasSgemm(cublas_handle,
				CUBLAS_OP_T, CUBLAS_OP_N,
				output_dim, effective_batch, input_dim,
				&_one,
				tensor_mem(linear->w), input_dim,
				tensor_mem(x), input_dim,
				&_zero,
				tensor_mem(y), output_dim));

	if (linear->has_bias) {
		tensor_add(linear->b, y);
	}

	linear->x = x;
	linear->y = y;
	return y;
}

tensor_t* linear_inference(linear_t *linear, tensor_t *x)
{
	// same as training
	return linear_forward(linear, x);
}

tensor_t* linear_backward(linear_t *linear, tensor_t *dy)
{
	// dy : batch_size x output_dim
	//
	// dx : batch_size x input_dim

	int output_dim = linear->output_dim;
	int input_dim = linear->input_dim;
	int effective_batch = tensor_nelem(dy) / output_dim;

	tensor_t *dx = tensor_samesize(linear->x);
	tstack_push(linear->tstack, dx);

	chkCUBLAS(cublasSgemm(cublas_handle,
				CUBLAS_OP_N, CUBLAS_OP_N,
				input_dim, effective_batch, output_dim,
				&_one,
				tensor_mem(linear->w), input_dim,
				tensor_mem(dy), output_dim,
				&_zero, tensor_mem(dx), input_dim));
	linear->dx = dx;

	chkCUBLAS(cublasSgemm(cublas_handle,
				CUBLAS_OP_N, CUBLAS_OP_T,
				input_dim, output_dim, effective_batch,
				&_one,
				tensor_mem(linear->x), input_dim,
				tensor_mem(dy), output_dim,
				&_one,
				tensor_mem(linear->dw), input_dim));

	if (linear->has_bias) {
		int nelem = tensor_nelem(dy);
		int n = tensor_size(dy, tensor_dim(dy)-1);
		int sizes[] = { nelem / n, n };
		tensor_view(dy, sizes, 2);
		tensor_t *db = tensor_reduce_sum(dy, 0);
		tensor_unsqueeze(db, 0);
		tstack_push(linear->tstack, db);
		tensor_add(db, linear->db);
	}

	return dx;
}

size_t linear_params(linear_t *linear)
{
	size_t sum = linear->input_dim * linear->output_dim;

	if (linear->has_bias)
		sum += linear->output_dim;

	return sum;
}

size_t linear_init_params(linear_t *linear, float *param, float *dparam)
{
	tensor_set_mem(linear->w, param);
	tensor_set_mem(linear->dw, dparam);
	snudnn_uniform(tensor_mem(linear->w),
			linear->input_dim * linear->output_dim, -0.1, 0.1);

	size_t offset = linear->input_dim * linear->output_dim;
	if (linear->has_bias) {
		tensor_set_mem(linear->b, param+offset);
		tensor_set_mem(linear->db, dparam+offset);
		snudnn_uniform(tensor_mem(linear->b),
				linear->output_dim, -0.1, 0.1);
		offset += linear->output_dim;
	}
	return offset;
}

void linear_clear(linear_t *linear)
{
	tstack_clear(linear->tstack);
}

void linear_zerograd(linear_t *linear)
{
	chkCUDA(cudaMemset(tensor_mem(linear->dw), 0, sizeof(float) * tensor_nelem(linear->dw)));
}

void linear_update(linear_t *linear, int N)
{
	optimizer_step(tensor_mem(linear->dw),
			tensor_mem(linear->w),
			linear->input_dim * linear->output_dim,
			N);

	if (linear->has_bias) {
		optimizer_step(tensor_mem(linear->db),
				tensor_mem(linear->b),
				linear->output_dim,
				N);
	}
}
