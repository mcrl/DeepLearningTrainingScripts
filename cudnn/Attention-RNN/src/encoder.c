#include <assert.h>
#include <stdio.h>
#include <stdlib.h>

#include "encoder.h"
#include "embedding.h"
#include "options.h"
#include "gpu.h"
#include "lstm.h"
#include "tstack.h"

struct _encoder_t {
	struct encoder_options *opts;

	embedding_t *embedding;
	dropout_t *dropout_in;
	dropout_t *dropout_out;
	lstm_t *lstm;


	struct tstack *tstack;
};

static void encoder_init(encoder_t *encoder, struct encoder_options *opts)
{
	encoder->opts = opts;

	encoder->embedding = embedding_create(
								opts->embedding.num_embeddings, 
								opts->embedding.embed_dim,
								opts->embedding.padding_idx
								);

	encoder->dropout_in = dropout_create(opts->dropout.in);

	encoder->lstm = lstm_create(opts->embedding.embed_dim,
								opts->lstm.hidden_size,
								opts->lstm.layers,
								opts->dropout.out
								);

	encoder->dropout_out = dropout_create(opts->dropout.out);

	encoder->tstack = tstack_create(16);
}

static void encoder_release(encoder_t *encoder)
{
	embedding_free(encoder->embedding);
	dropout_free(encoder->dropout_in);
	lstm_free(encoder->lstm);
	dropout_free(encoder->dropout_out);
}

encoder_t* encoder_create(struct encoder_options *opts)
{
	encoder_t *encoder = malloc(sizeof(encoder_t));
	encoder_init(encoder, opts);
	return encoder;
}

void encoder_free(encoder_t *encoder)
{
	encoder_release(encoder);
	free(encoder);
}

tensor_quadruple_t encoder_forward(encoder_t *encoder, tensor_t *src_toks, const int *seq_len_array)
{
	tensor_t *embedding = embedding_forward(encoder->embedding, src_toks);

	tensor_t *dropout_in = dropout_forward(encoder->dropout_in, embedding);

	tensor_t *dropout_in_tr = tensor_transpose(dropout_in, 0, 1);

	assert(!encoder->opts->lstm.bidirectional);

	tensor_triple_t lstm_out = lstm_forward(encoder->lstm, dropout_in_tr, NULL, NULL, seq_len_array);

	tensor_t *dropout_out = dropout_forward(encoder->dropout_out, lstm_out.first);

	tensor_t *encoder_padding_mask = tensor_mask(src_toks, encoder->opts->embedding.padding_idx);
	tensor_quadruple_t out = { .first  = dropout_out,
							.second = lstm_out.second,
							.third  = lstm_out.third ,
							.fourth = encoder_padding_mask
							};

	tstack_push(encoder->tstack, dropout_in_tr);
	tstack_push(encoder->tstack, encoder_padding_mask);
	return out;
}

tensor_quadruple_t encoder_inference(encoder_t *encoder, tensor_t *src_toks, const int *seq_len_array)
{
	tensor_t *embedding = embedding_inference(encoder->embedding, src_toks);

	tensor_t *dropout_in = dropout_inference(encoder->dropout_in, embedding);

	tensor_t *dropout_in_tr = tensor_transpose(dropout_in, 0, 1);

	assert(!encoder->opts->lstm.bidirectional);

	tensor_triple_t lstm_out = lstm_inference(encoder->lstm, dropout_in_tr, NULL, NULL, seq_len_array);

	tensor_t *dropout_out = dropout_inference(encoder->dropout_out, lstm_out.first);

	tensor_t *encoder_padding_mask = tensor_mask(src_toks, encoder->opts->embedding.padding_idx);
	tensor_quadruple_t out = { .first  = dropout_out,
							.second = lstm_out.second,
							.third  = lstm_out.third ,
							.fourth = encoder_padding_mask
							};

	tstack_push(encoder->tstack, dropout_in_tr);
	tstack_push(encoder->tstack, encoder_padding_mask);
	return out;
}

tensor_t* encoder_backward(encoder_t *encoder, tensor_t *dy, tensor_t *dhy, tensor_t *dcy)
{
	tensor_t *dx_dropout_out = dropout_backward(encoder->dropout_out, dy);

	tensor_triple_t dx_lstm = lstm_backward(encoder->lstm, dx_dropout_out, dhy, dcy);

	tensor_t *dy_dropout_in = tensor_transpose(dx_lstm.first, 0, 1);

	tensor_t *dy_embedding = dropout_backward(encoder->dropout_in, dy_dropout_in);

	tensor_t *dx = embedding_backward(encoder->embedding, dy_embedding);
	assert(dx == NULL);

	tstack_push(encoder->tstack, dy_dropout_in);
	return dx;
}

size_t encoder_params(encoder_t *encoder)
{
	size_t sum = 0;
	sum += embedding_params(encoder->embedding);
	sum += dropout_params(encoder->dropout_in);
	sum += dropout_params(encoder->dropout_out);
	sum += lstm_params(encoder->lstm);
	return sum;
}

size_t encoder_init_params(encoder_t *encoder, float *param, float *dparam)
{
	size_t offset = 0;
	offset += embedding_param_init(encoder->embedding, param + offset, dparam + offset);
	offset += dropout_init_params(encoder->dropout_in, param + offset, dparam + offset);
	offset += dropout_init_params(encoder->dropout_out, param + offset, dparam + offset);
	offset += lstm_param_init(encoder->lstm, param + offset, dparam + offset);
	return offset;
}

void encoder_clear(encoder_t *encoder)
{
	dropout_clear(encoder->dropout_out);
	embedding_clear(encoder->embedding);
	dropout_clear(encoder->dropout_in);
	lstm_clear(encoder->lstm);
	tstack_clear(encoder->tstack);
}

void encoder_zerograd(encoder_t *encoder)
{
	embedding_zerograd(encoder->embedding);
	dropout_zerograd(encoder->dropout_in);
	dropout_zerograd(encoder->dropout_out);
	lstm_zerograd(encoder->lstm);
}

void encoder_update(encoder_t *encoder, int N)
{
	embedding_update(encoder->embedding, N);
	dropout_update(encoder->dropout_in, N);
	dropout_update(encoder->dropout_out, N);
	lstm_update(encoder->lstm, N);
}
