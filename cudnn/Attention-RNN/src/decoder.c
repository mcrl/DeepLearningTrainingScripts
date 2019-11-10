#include <stdio.h>
#include <stdlib.h>
#include "decoder.h"
#include "options.h"
#include "linear.h"
#include "gpu.h"
#include "lstm_attn.h"
#include "utils.h"
#include "tstack.h"

struct _decoder_t {
	struct decoder_options opts;

	embedding_t *embedding;
	dropout_t *dropout_in;

	lstm_attn_t *lstm_attn;

	bool has_additional_fc;
	linear_t *additional_fc;
	dropout_t *additional_dropout;

	linear_t *fc_out;
	dropout_t *dropout_out;


	struct tstack *tstack;
};

static void decoder_init(decoder_t *d, struct decoder_options opts)
{
	d->opts = opts;
	d->embedding = embedding_create(opts.embedding.num_embeddings,
			opts.embedding.embed_dim,
			opts.embedding.padding_idx);

	d->dropout_in = dropout_create(opts.dropout.in);


	d->lstm_attn = lstm_attn_create(
			opts.embedding.embed_dim,
			opts.lstm.hidden_size,
			opts.lstm.layers,
			opts.dropout.out,
			opts.lstm.bidirectional,
			opts.lstm.max_len,
			opts.embedding.padding_idx
			);

	d->has_additional_fc = opts.out_embed_dim != opts.lstm.hidden_size;
	if (d->has_additional_fc) {
		d->additional_fc = linear_create(opts.lstm.hidden_size, 
				opts.out_embed_dim,
				true);
		d->additional_dropout = dropout_create(opts.dropout.out);
	}

	d->tstack = tstack_create(16);

	d->fc_out = linear_create(opts.out_embed_dim, opts.embedding.num_embeddings, true);

	d->dropout_out = dropout_create(opts.dropout.out);
}

static void decoder_release(decoder_t *d)
{
	embedding_free(d->embedding);
	if (d->has_additional_fc) {
		linear_free(d->additional_fc);
	}
}

decoder_t* decoder_create(struct decoder_options opts)
{
	decoder_t *decoder = malloc(sizeof(decoder_t));
	decoder_init(decoder, opts);
	return decoder;
}

void decoder_free(decoder_t *decoder)
{
	decoder_release(decoder);
	free(decoder);
}

tensor_t* decoder_forward(decoder_t *decoder,
		tensor_t *prev_output_tokens,
		tensor_t *encoder_outs,
		tensor_t *encoder_hiddens,
		tensor_t *encoder_cells,
		tensor_t *encoder_padding_mask)
{
	// prev_output_tokens: batch_size x tgt_len (int)
	// encoder_outs: src_len x batch_size x hidden_size
	// encoder_hiddens: num_layers x batch_size x hidden_size
	// encoder_cells: num_layers x batch_size x hidden_size
	// encoder_padding_mask: batch_size x src_len

	//-> batch_size x tgt_len x embed_dim
	tensor_t *embedding;
	tensor_t *dropout_in;
	tensor_t *dropout_in_tr;
	START_TIMER_ {
		embedding = embedding_forward(decoder->embedding, prev_output_tokens);
		dropout_in = dropout_forward(decoder->dropout_in, embedding);

		//-> tgt_len x batch_size x embed_dim
		dropout_in_tr = tensor_transpose(dropout_in, 0, 1); 
	} STOP_TIMER_ ("decoder_forward_init")


	// Initialize previous states
	tensor_t *lstm_attn;
	START_TIMER_ {
		lstm_attn = lstm_attn_forward(decoder->lstm_attn,
				dropout_in_tr,
				encoder_outs,
				encoder_hiddens,
				encoder_cells,
				encoder_padding_mask);
	} STOP_TIMER_ ("decoder_forward_attn")

	tensor_t *out;
	START_TIMER_ {
		//-> batch_size x tgt_len x hidden_size
		tensor_t *lstm_attn_tr;
		START_TIMER_ {
			lstm_attn_tr = tensor_transpose(lstm_attn, 0, 1); 
		} STOP_TIMER_ ("decoder_forward_left_tr")

		tensor_t *out_proj = lstm_attn_tr;
		if (decoder->has_additional_fc) {
			//-> batch_size x tgt_len x out_embed_dim
			tensor_t *additional_fc = linear_forward(decoder->additional_fc, lstm_attn_tr);
			tensor_t *additional_dropout = dropout_forward(decoder->additional_dropout, additional_fc);
			out_proj = additional_dropout;
		}

		//-> batch_size x tgt_len x num_embeddings
		tensor_t *fc_out;
		START_TIMER_ {
			fc_out = linear_forward(decoder->fc_out, out_proj);
		} STOP_TIMER_ ("decoder_forward_left_linear")

		out = dropout_forward(decoder->dropout_out, fc_out);
		tstack_push(decoder->tstack, dropout_in_tr);
		tstack_push(decoder->tstack, lstm_attn_tr);

	} STOP_TIMER_ ("decoder_forward_left")

	return out;
}

tensor_t* decoder_inference(decoder_t *decoder,
		tensor_t *prev_output_tokens,
		tensor_t *encoder_outs,
		tensor_t *encoder_hiddens,
		tensor_t *encoder_cells,
		tensor_t *encoder_padding_mask)
{
	tensor_t *embedding;
	tensor_t *dropout_in;
	tensor_t *dropout_in_tr;
	embedding = embedding_inference(decoder->embedding, prev_output_tokens);
	dropout_in = dropout_inference(decoder->dropout_in, embedding);
	//-> tgt_len x batch_size x embed_dim
	dropout_in_tr = tensor_transpose(dropout_in, 0, 1); 


	// Initialize previous states
	tensor_t *lstm_attn;
	lstm_attn = lstm_attn_inference(decoder->lstm_attn,
			dropout_in_tr,
			encoder_outs,
			encoder_hiddens,
			encoder_cells,
			encoder_padding_mask);

	//-> batch_size x tgt_len x hidden_size
	tensor_t *lstm_attn_tr = tensor_transpose(lstm_attn, 0, 1); 

	tensor_t *out_proj = lstm_attn_tr;
	if (decoder->has_additional_fc) {
		//-> batch_size x tgt_len x out_embed_dim
		tensor_t *additional_fc = linear_inference(decoder->additional_fc, lstm_attn_tr);
		tensor_t *additional_dropout = dropout_inference(decoder->additional_dropout, additional_fc);
		out_proj = additional_dropout;
	}

	//-> batch_size x tgt_len x num_embeddings
	tensor_t *fc_out;
	tensor_t *out;
	fc_out = linear_inference(decoder->fc_out, out_proj);
	out = dropout_inference(decoder->dropout_out, fc_out);

	tstack_push(decoder->tstack, dropout_in_tr);
	tstack_push(decoder->tstack, lstm_attn_tr);

	return out;
}


tensor_triple_t decoder_backward(decoder_t *decoder, tensor_t *dy)
{
	// dy : batch_size x tgt_len x num_embeddings
	tensor_t *dy_fc_out = dropout_backward(decoder->dropout_out, dy);


	tensor_t *dx_fc_out;
	dx_fc_out = linear_backward(decoder->fc_out, dy_fc_out);
	//-> batch_size x tgt_len x out_embed_dim

	tensor_t *dx_proj = dx_fc_out;
	if (decoder->has_additional_fc) {
		//-> batch_size x tgt_len x hidden_size
		tensor_t *dy_fc = dropout_backward(decoder->additional_dropout, dx_fc_out);
		dx_proj = linear_backward(decoder->additional_fc, dy_fc);
	} 

	//-> tgt_len x batch_size x hidden_size
	tensor_t *dy_lstm_attn = tensor_transpose(dx_proj, 0, 1);

	tensor_quadruple_t quad;
	quad = lstm_attn_backward(decoder->lstm_attn, dy_lstm_attn);

	tensor_t *dy_in = quad.first;       //-> tgt_len x batch_size x embed_dim
	tensor_t *dx_out = quad.second;     //-> src_len x batch_size x hidden_size
	tensor_t *dx_hiddens = quad.third;  //-> num_layers x batch_size x hidden_size
	tensor_t *dx_cells = quad.fourth;   //-> num_layers x batch_size x hidden_size

	//-> batch_size x tgt_len x embed_dim
	tensor_t *dy_dropout_in = tensor_transpose(dy_in, 0, 1);

	tensor_t *dy_embedding = dropout_backward(decoder->dropout_in, dy_dropout_in);

	//-> batch_size x tgt_len
	//tensor_t *dx = embedding_backward(decoder->embedding, dy_embedding);
	embedding_backward(decoder->embedding, dy_embedding);

	tensor_triple_t dx_triple = { 
		.first = dx_out,
		.second = dx_hiddens,
		.third = dx_cells
	};

	tstack_push(decoder->tstack, dy_lstm_attn);
	tstack_push(decoder->tstack, dy_dropout_in);

	return dx_triple;
}

size_t decoder_params(decoder_t *decoder)
{
	size_t sum = 0;
	sum += embedding_params(decoder->embedding);
	sum += dropout_params(decoder->dropout_in);

	sum += lstm_attn_params(decoder->lstm_attn);

	if (decoder->has_additional_fc) {
		sum += linear_params(decoder->additional_fc);
		sum += dropout_params(decoder->additional_dropout);
	}

	sum += linear_params(decoder->fc_out);
	sum += dropout_params(decoder->dropout_out);
	return sum;
}

size_t decoder_init_params(decoder_t *decoder, float *param, float *dparam)
{
	size_t offset = 0;
	offset += embedding_param_init(decoder->embedding, param+offset, dparam+offset);
	offset += dropout_init_params(decoder->dropout_in, param+offset, dparam+offset);

	offset += lstm_attn_param_init(decoder->lstm_attn, param+offset, dparam+offset);

	if (decoder->has_additional_fc) {
		offset += linear_init_params(decoder->additional_fc, param+offset, dparam+offset);
		offset += dropout_init_params(decoder->additional_dropout, param+offset, dparam+offset);
	}

	offset += linear_init_params(decoder->fc_out, param+offset, dparam+offset);
	offset += dropout_init_params(decoder->dropout_out, param+offset, dparam+offset);
	return offset;
}

void decoder_clear(decoder_t *decoder)
{
	embedding_clear(decoder->embedding);
	dropout_clear(decoder->dropout_in);
	lstm_attn_clear(decoder->lstm_attn);

	if (decoder->has_additional_fc) {
		linear_clear(decoder->additional_fc);
		dropout_clear(decoder->additional_dropout);
	}
	linear_clear(decoder->fc_out);
	dropout_clear(decoder->dropout_out);

	tstack_clear(decoder->tstack);
}

void decoder_zerograd(decoder_t *decoder)
{
	embedding_zerograd(decoder->embedding);
	dropout_zerograd(decoder->dropout_in);

	lstm_attn_zerograd(decoder->lstm_attn);
	if (decoder->has_additional_fc) {
		linear_zerograd(decoder->additional_fc);
		dropout_zerograd(decoder->additional_dropout);
	}
	linear_zerograd(decoder->fc_out);
	dropout_zerograd(decoder->dropout_out);
}

void decoder_update(decoder_t *decoder, int N)
{
	embedding_update(decoder->embedding, N);
	dropout_update(decoder->dropout_in, N);

	lstm_attn_update(decoder->lstm_attn, N);
	if (decoder->has_additional_fc) {
		linear_update(decoder->additional_fc, N);
		dropout_update(decoder->additional_dropout, N);
	}
	linear_update(decoder->fc_out, N);
	dropout_update(decoder->dropout_out, N);
}
