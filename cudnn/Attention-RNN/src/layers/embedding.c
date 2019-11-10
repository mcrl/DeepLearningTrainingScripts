#include <stdlib.h>
#include <assert.h>

#include "embedding.h"
#include "gpu.h"
#include "snudnn.h"
#include "optimizer.h"

void cuda_embedding_forward(const float *w, const int *indicies, float *y, int N, int embed_dim);
void cuda_embedding_backward(const float *dy, const int *indicies, float *dw, int N, int embed_dim);

struct _embedding_t {
	int num_embeddings;
	int embed_dim;
	int padding_idx;

	tensor_t *x;
	tensor_t *y;
	float *w; // num_embeddings x embed_dim
	float *dw;
};

static void embedding_init(embedding_t *embedding, int num_embeddings, int embed_dim, int padding_idx)
{
	embedding->num_embeddings = num_embeddings;
	embedding->embed_dim = embed_dim;
	embedding->padding_idx = padding_idx;
}

static void embedding_release(embedding_t *embedding)
{
}

size_t embedding_params(embedding_t *embedding)
{
	return embedding->num_embeddings * embedding->embed_dim;
}

size_t embedding_param_init(embedding_t *embedding, float *param, float *dparam)
{
	embedding->w = param;
	embedding->dw = dparam;
	snudnn_uniform(embedding->w, embedding->num_embeddings * embedding->embed_dim, -0.1, 0.1);

	// Set 'padding_idx' row to zero
	int padding_idx_start = embedding->padding_idx * embedding->embed_dim;
	chkCUDA(cudaMemset(&embedding->w[padding_idx_start], 0, sizeof(float) * embedding->embed_dim));
	return embedding->num_embeddings * embedding->embed_dim;
}

void embedding_clear(embedding_t *embedding)
{
	tensor_free(embedding->y);
}

void embedding_clean(embedding_t *embedding)
{
	tensor_free(embedding->y);
}

embedding_t* embedding_create (int num_embeddings, int embed_dim, int padding_idx)
{
	embedding_t *embedding = malloc(sizeof(embedding_t));
	embedding_init(embedding, num_embeddings, embed_dim, padding_idx);
	return embedding;
}

void embedding_free(embedding_t *embedding)
{
	embedding_release(embedding);
	free(embedding);
}

tensor_t* embedding_forward(embedding_t *embedding, tensor_t *x)
{
	// (in)
	// 		x : batch_size x seq_len
	// (out)
	// 		y : batch_size x seq_len x embed_dim

	int batch_size = tensor_size(x, 0);
	int seq_len = tensor_size(x, 1);
	int embed_dim = embedding->embed_dim;

	int sizes[] = { batch_size, seq_len, embed_dim };
	tensor_t *y = tensor_create(sizes, 3);

	cuda_embedding_forward(
			embedding->w,
			tensor_mem(x),
			tensor_mem(y),
			batch_size * seq_len,
			embed_dim);

	embedding->x = x;
	embedding->y = y;

	return y;
}

tensor_t* embedding_inference(embedding_t *embedding, tensor_t *x)
{
	// same as training
	return embedding_forward(embedding, x);
}

tensor_t* embedding_backward(embedding_t *embedding, tensor_t *dy)
{
	// (in)
	// 		dy : batch_size x seq_len x embed_dim
	// (out)
	// 		dw : num_embeddings x embed_dim
	// 		dx : null

	int batch_size = tensor_size(embedding->x, 0);
	int seq_len = tensor_size(embedding->x, 1);
	int embed_dim = embedding->embed_dim;

	cuda_embedding_backward(tensor_mem(dy),
							tensor_mem(embedding->x),
							embedding->dw,
							batch_size * seq_len,
							embed_dim);

	// Nothing to returns
	return NULL;
}

void embedding_zerograd(embedding_t *embedding)
{
	int nelem = embedding->num_embeddings * embedding->embed_dim;
	chkCUDA(cudaMemset(embedding->dw, 0, nelem * sizeof(float)));
}

void embedding_update(embedding_t *embedding, int N)
{
	optimizer_step(
			embedding->dw,
			embedding->w,
			embedding->embed_dim * embedding->num_embeddings,
			N);
}
