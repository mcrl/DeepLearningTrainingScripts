#include "loss.h"
#include "model.h"
#include "kernels.h"

#include "gpu.h"
#include "snudnn.h"
#include "utils.h"
#include <assert.h>
#include <stdlib.h>


void cuda_gather_2d(const float *in, float *out, const int *index, int N, int M);
void cuda_softmax_dy(const int *index, float *dy, int N, int M);
static loss_info_t cross_entropy_forward(loss_t *loss, tensor_t *out, tensor_t *target);
static tensor_t* cross_entropy_backward(loss_t *loss, tensor_t *out, tensor_t *target);
static loss_info_t cross_entropy_inference(loss_t *loss, tensor_t *out, tensor_t *target);
// -------------------------------------------------------------

struct _loss_t {
	loss_info_t (*alg_forward)(loss_t *loss, tensor_t *out, tensor_t *target);
	loss_info_t (*alg_inference)(loss_t *loss, tensor_t *out, tensor_t *target);
	tensor_t* (*alg_backward)(loss_t *loss, tensor_t *out, tensor_t *target);


	model_t* model;
	tensor_t* softmax;
};


static void loss_init(loss_t *loss, model_t *model)
{
	loss->model = model;
	loss->alg_forward = cross_entropy_forward;
	loss->alg_inference = cross_entropy_inference;
	loss->alg_backward = cross_entropy_backward;
}

loss_t* loss_create(model_t *model)
{
	loss_t *loss = malloc(sizeof(loss_t));
	loss_init(loss, model);
	return loss;
}

void loss_free(model_t *model)
{
	free(model);
}

loss_info_t loss_inference(loss_t *loss, tensor_t *out, tensor_t *target)
{
	// out: batch_size x seq_len x num_embeddings
	// target: batch_size x seq_len x num_embeddings
	return loss->alg_inference(loss, out, target);
}

loss_info_t loss_forward(loss_t *loss, tensor_t *out, tensor_t *target)
{
	// out: batch_size x seq_len x num_embeddings
	// target: batch_size x seq_len
	return loss->alg_forward(loss, out, target);
}

tensor_t* loss_backward(loss_t *loss, tensor_t *out, tensor_t *target)
{
	tensor_t *dy = loss->alg_backward(loss, out, target);
	model_backward(loss->model, dy);
	tensor_free(dy);
	return dy;
}
  


// -------------------------------------------------------------
static loss_info_t cross_entropy_forward(loss_t *loss, tensor_t *out, tensor_t *target)
{
	loss_info_t info;

	// out: batch_size x seq_len x num_embeddings
	// target: batch_size x seq_len

	assert(tensor_type(target) == TENSOR_INT);
	assert(tensor_size(out, 0) == tensor_size(target, 0));
	assert(tensor_size(out, 1) == tensor_size(target, 1));

	int batch_size = tensor_size(out, 0);
	int seq_len = tensor_size(out, 1);
	int num_embeddings = tensor_size(out, 2);

	{
		int sizes[] = { batch_size * seq_len, num_embeddings };
		tensor_view(out, sizes, 2);
	}
	tensor_flatten(target);

	tensor_t *softmax = tensor_samesize(out);
	loss->softmax = softmax;
	chkCUDNN(cudnnSoftmaxForward(cudnn_handle,
				CUDNN_SOFTMAX_LOG,
				CUDNN_SOFTMAX_MODE_INSTANCE,
				&_one, tensor_descriptor(out), tensor_mem(out),
				&_zero, tensor_descriptor(softmax), tensor_mem(softmax)));

	int sizes[] = { tensor_size(target, 0) };
	tensor_t *gather = tensor_create(sizes, 1);
	cuda_gather_2d(tensor_mem(softmax), tensor_mem(gather), tensor_mem(target), tensor_size(out, 0), tensor_size(out, 1));

	int nelem = batch_size * seq_len;
	float loss_sum = tensor_asum(gather);
	info.nll_loss = loss_sum / nelem;
	info.loss = info.nll_loss;
	{
		int sizes[] = { batch_size, seq_len, num_embeddings };
		tensor_view(out, sizes, 3);
	}
	{
		int sizes[] = { batch_size, seq_len };
		tensor_view(target, sizes, 2);
	}

	return info;
}

static loss_info_t cross_entropy_inference(loss_t *loss, tensor_t *out, tensor_t *target)
{
	loss_info_t info;

	// out: batch_size x seq_len x num_embeddings
	// target: batch_size x seq_len

	assert(tensor_type(target) == TENSOR_INT);
	assert(tensor_size(out, 0) == tensor_size(target, 0));
	assert(tensor_size(out, 1) == tensor_size(target, 1));

	int batch_size = tensor_size(out, 0);
	int seq_len = tensor_size(out, 1);
	int num_embeddings = tensor_size(out, 2);

	{
		int sizes[] = { batch_size * seq_len, num_embeddings };
		tensor_view(out, sizes, 2);
	}
	tensor_flatten(target);

	tensor_t *softmax = tensor_samesize(out);
	loss->softmax = softmax;
	chkCUDNN(cudnnSoftmaxForward(cudnn_handle,
				CUDNN_SOFTMAX_LOG,
				CUDNN_SOFTMAX_MODE_INSTANCE,
				&_one, tensor_descriptor(out), tensor_mem(out),
				&_zero, tensor_descriptor(softmax), tensor_mem(softmax)));

	int sizes[] = { tensor_size(target, 0) };
	tensor_t *gather = tensor_create(sizes, 1);
	cuda_gather_2d(tensor_mem(softmax), tensor_mem(gather), tensor_mem(target), tensor_size(out, 0), tensor_size(out, 1));

	int nelem = batch_size * seq_len;
	float loss_sum = tensor_asum(gather);
	info.nll_loss = loss_sum / nelem;
	info.loss = info.nll_loss;
	{
		int sizes[] = { batch_size, seq_len, num_embeddings };
		tensor_view(out, sizes, 3);
	}
	{
		int sizes[] = { batch_size, seq_len };
		tensor_view(target, sizes, 2);
	}

	tensor_free(softmax);
	tensor_free(gather);


	return info;
}

static tensor_t* cross_entropy_backward(loss_t *loss, tensor_t *out, tensor_t *target)
{
	// out: batch_size x seq_len x num_embeddings
	// target: batch_size x seq_len
	assert(tensor_type(target) == TENSOR_INT);
	assert(tensor_size(out, 0) == tensor_size(target, 0));
	assert(tensor_size(out, 1) == tensor_size(target, 1));

	int batch_size = tensor_size(out, 0);
	int seq_len = tensor_size(out, 1);
	int num_embeddings = tensor_size(out, 2);
	
	tensor_t *dy = tensor_zeros(tensor_sizes(out), tensor_dim(out));
	cuda_softmax_dy(tensor_mem(target), tensor_mem(dy), batch_size * seq_len, num_embeddings);
	{
		int sizes[] = { batch_size * seq_len, num_embeddings };
		tensor_view(dy, sizes, 2); 
		tensor_view(out, sizes, 2); 
	}


	chkCUDNN(cudnnSoftmaxBackward(cudnn_handle,
				CUDNN_SOFTMAX_LOG,
				CUDNN_SOFTMAX_MODE_INSTANCE,
				&_one, tensor_descriptor(loss->softmax), tensor_mem(loss->softmax),
				tensor_descriptor(dy), tensor_mem(dy),
				&_zero, tensor_descriptor(dy), tensor_mem(dy)));
	{
		int sizes[] = { batch_size,  seq_len, num_embeddings };
		tensor_view(dy, sizes, 3); 
		tensor_view(out, sizes, 3); 
	}
	tensor_free(loss->softmax);


	return dy;
}
