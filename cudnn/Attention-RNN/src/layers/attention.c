#include <assert.h>
#include <stdlib.h>
#include <float.h>
#include "attention.h"
#include "gpu.h"
#include "linear.h"
#include "snudnn.h"
#include "tensor.h"
#include "tstack.h"
#include "utils.h"

void cuda_masked_copy(float *y, const int *mask, float val, int N);

struct _attention_t {
	int input_embed_dim;
	int source_embed_dim;
	int output_embed_dim;
	bool bias;

	int prev_dim[2];

	linear_t *input_proj;
	linear_t *output_proj;


	struct tstack *attn_scores_st;
	struct tstack *tanh_x_st;
	struct tstack *tanh_y_st;


	struct tstack *source_hids_st;
	struct tstack *input_proj_y_st;

	tensor_t *encoder_padding_mask;

	struct tstack *tstack;
};

static void attention_init(attention_t *attention, int input_embed_dim, int source_embed_dim, int output_embed_dim, bool bias)
{
	attention->input_embed_dim = input_embed_dim;
	attention->source_embed_dim = source_embed_dim;
	attention->output_embed_dim = output_embed_dim;
	attention->bias = bias;

	attention->input_proj = linear_create(input_embed_dim, source_embed_dim, bias);
	attention->output_proj = linear_create(source_embed_dim + input_embed_dim, output_embed_dim, bias);

	attention->tstack = tstack_create(16);
	attention->attn_scores_st = tstack_create(16);
	attention->tanh_x_st = tstack_create(16);
	attention->tanh_y_st = tstack_create(16);
	attention->source_hids_st = tstack_create(16);
	attention->input_proj_y_st = tstack_create(16);
}

static void attention_release(attention_t *attention)
{
}

attention_t *attention_create(int input_embed_dim, int source_embed_dim, int output_embed_dim, bool bias)
{
	attention_t *attention = malloc(sizeof(attention_t));
	attention_init(attention, input_embed_dim, source_embed_dim, output_embed_dim, bias);
	return attention;
}

void attention_free(attention_t *attention)
{
	attention_release(attention);
	free(attention);
}

tensor_t* attention_forward(attention_t *attention, tensor_t *input, tensor_t *source_hids, tensor_t *encoder_padding_mask)
{
	// input: batch_size x input_embed_dim
	// source_hids: src_len x batch_size x source_embed_dim
	// encoder_padding_mask: batch_size x src_len
	// 
	// tanh: batch_size x output_embed_dim
	// attn_scores: seqlen x batch_size

	int batch_size = tensor_size(input, 0);
	assert(batch_size == tensor_size(source_hids, 1));


	//-> batch_size x source_embed_dim
	tensor_t *input_proj_y;
	input_proj_y = linear_forward(attention->input_proj, input);

	tensor_t *pw_mult;
	tensor_t *rsum;
	tensor_unsqueeze(input_proj_y, 0);
	pw_mult  = tensor_pointwise_mult(source_hids, input_proj_y);
	tensor_squeeze(input_proj_y, 0); 
	tstack_push(attention->tstack, pw_mult);

	//-> src_len x batch_size
	attention->prev_dim[0] = tensor_size(pw_mult, 2);
	rsum  = tensor_reduce_sum(pw_mult, 2);
	tstack_push(attention->tstack, rsum);

	if (encoder_padding_mask) {
		cuda_masked_copy(tensor_mem(rsum), tensor_mem(encoder_padding_mask),
				-FLT_MAX, tensor_nelem(rsum));
	}

	//-> src_len x batch_size
	tensor_t *attn_scores;
	attn_scores = snudnn_softmax0_forward(rsum);

	//-> src_len x batch_size x source_embed_dim
	tensor_t *reduced_sum;
	tensor_unsqueeze(attn_scores, 2);
	tensor_t *mult;
	mult = tensor_pointwise_mult(source_hids, attn_scores);
	tensor_squeeze(attn_scores, 2);
	tstack_push(attention->tstack, mult);

	//-> batch_size x source_embed_dim
	attention->prev_dim[1] = tensor_size(mult, 0);
	reduced_sum = tensor_reduce_sum(mult, 0);
	tstack_push(attention->tstack, reduced_sum);

	//-> batch_size x (source_embed_dim + input_embed_dim)
	tensor_t *cat;
	tensor_t *output_proj;
	tensor_t *tanh;
	cat = tensor_concat(reduced_sum, input, 1);
	tstack_push(attention->tstack, cat);

	//-> batch_size x output_embed_dim

	output_proj = linear_forward(attention->output_proj, cat);
	tanh = snudnn_tanh_forward(output_proj);

	tstack_push(attention->attn_scores_st, attn_scores);
	tstack_push(attention->tanh_x_st, output_proj);
	tstack_push(attention->tanh_y_st, tanh);
	tstack_push(attention->input_proj_y_st, input_proj_y);
	tstack_push(attention->source_hids_st, source_hids);
	attention->encoder_padding_mask = encoder_padding_mask;

	return tanh;
}

tensor_t* attention_inference(attention_t *attention, tensor_t *input, tensor_t *source_hids, tensor_t *encoder_padding_mask)
{
}

tensor_pair_t attention_backward(attention_t *attention, tensor_t *dy)
{
	// dy: batch_size x output_embed_dim
	int batch_size = tensor_size(dy, 0);

	tensor_t *dx_tanh = snudnn_tanh_backward(dy,
			tstack_top(attention->tanh_x_st),
			tstack_top(attention->tanh_y_st));
	tstack_push(attention->tstack, dx_tanh);

	//-> batch_size x (input_embed_dim + source_embed_dim)
	tensor_t *dx_output_proj = linear_backward(attention->output_proj, dx_tanh);

	//->  batch_size x source_embed_dim // batch_size x input_embed_dim
	int source_embed_dim = attention->source_embed_dim;
	int sizes[] = { batch_size, source_embed_dim };
	tensor_pair_t tpair = tensor_split(dx_output_proj, sizes, 1);
	tensor_t *dx_reduced_sum = tpair.first;             // batch_size x source_embed_dim
	tensor_t *dx_input1 = tpair.second;                 // batch_size x input_embed_dim
	tstack_push(attention->tstack, dx_reduced_sum);
	tstack_push(attention->tstack, dx_input1);


	//-> src_len x batch_size x source_embed_dim
	tensor_t *dy_mult = tensor_expand_sum(dx_reduced_sum, 0, attention->prev_dim[1]);
	tstack_push(attention->tstack, dy_mult);

	tensor_unsqueeze(tstack_top(attention->attn_scores_st), 2);
	tensor_t *dx_source_hids1 = tensor_pointwise_mult(dy_mult, tstack_top(attention->attn_scores_st));
	tensor_t *dx_pw_mult = tensor_pointwise_mult(dy_mult, tstack_top(attention->source_hids_st));
	tensor_t *dy_softmax = tensor_reduce_sum(dx_pw_mult, 2);
	tensor_squeeze(tstack_top(attention->attn_scores_st), 2);
	tstack_push(attention->tstack, dx_source_hids1);
	tstack_push(attention->tstack, dx_pw_mult);
	tstack_push(attention->tstack, dy_softmax);

	//-> srclen x batch_size
	tensor_t *dy_rsum = snudnn_softmax0_backward(dy_softmax, tstack_top(attention->attn_scores_st));
	tstack_push(attention->tstack, dy_rsum);

	if (attention->encoder_padding_mask) {
		cuda_masked_copy(tensor_mem(dy_rsum),
				tensor_mem(attention->encoder_padding_mask),
				0, tensor_nelem(dy_rsum));
	}

	//-> srclen x batch_size x source_embed_dim
	tensor_t *dx_c = tensor_expand_sum(dy_rsum, 2, attention->prev_dim[0]);
	tstack_push(attention->tstack, dx_c);

	//-> src_len x batch_size x source_embed_dim
	tensor_unsqueeze(tstack_top(attention->input_proj_y_st), 0);
	tensor_t *dx_source_hids2 = tensor_pointwise_mult(dx_c, tstack_top(attention->input_proj_y_st));
	tensor_t *dx_pw_mult2 = tensor_pointwise_mult(dx_c, tstack_top(attention->source_hids_st));
	tensor_t *dy_input_proj = tensor_reduce_sum(dx_pw_mult2, 0);
	tensor_squeeze(tstack_top(attention->input_proj_y_st), 0);
	tstack_push(attention->tstack, dx_source_hids2);
	tstack_push(attention->tstack, dx_pw_mult2);
	tstack_push(attention->tstack, dy_input_proj);

	//-> batch_size x input_embed_dim
	tensor_t *dx_input2 = linear_backward(attention->input_proj, dy_input_proj);

	tensor_t *dx_source_hids = tensor_pointwise_add(dx_source_hids1, dx_source_hids2);
	tensor_t *dx_input = tensor_pointwise_add(dx_input1, dx_input2);


	tstack_push(attention->tstack, dx_input);
	tstack_push(attention->tstack, dx_source_hids);

	tstack_pop(attention->attn_scores_st);
	tstack_pop(attention->tanh_x_st);
	tstack_pop(attention->tanh_y_st);
	tstack_pop(attention->input_proj_y_st);
	tstack_pop(attention->source_hids_st);

	tensor_pair_t pair = {
		.first = dx_input,        // batch_size x input_embed_dim
		.second = dx_source_hids, // src_len x batch_size x source_embe_dim
	};
	return pair;
}

size_t attention_params(attention_t *attention)
{
	size_t sum = 0;
	sum += linear_params(attention->input_proj);
	sum += linear_params(attention->output_proj);
	return sum;
}

size_t attention_param_init(attention_t *attention, float *param, float *dparam)
{
	size_t offset = 0;
	offset += linear_init_params(attention->input_proj, param+offset, dparam+offset);
	offset += linear_init_params(attention->output_proj, param+offset, dparam+offset);
	return offset;
}

void attention_clear(attention_t *attention)
{
	linear_clear(attention->input_proj);
	linear_clear(attention->output_proj);

	tstack_clear(attention->tstack);
	tstack_clear(attention->attn_scores_st);
	tstack_clear(attention->tanh_y_st);
	tstack_clear_s(attention->tanh_x_st);
	tstack_clear_s(attention->source_hids_st);
	tstack_clear_s(attention->input_proj_y_st);
}

void attention_zerograd(attention_t *attention)
{
	linear_zerograd(attention->input_proj);
	linear_zerograd(attention->output_proj);
}

void attention_update(attention_t *attention, int N)
{
	linear_update(attention->input_proj, N);
	linear_update(attention->output_proj, N);
}
