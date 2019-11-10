#include <stdlib.h>
#include <stdbool.h>

#include "attention.h"
#include "lstm_attn.h"
#include "dropout.h"
#include "lstm_cell.h"
#include "linear.h"
#include "gpu.h"
#include "utils.h"
#include "tstack.h"

typedef struct _lstm_attn_t lstm_attn_t;

struct _lstm_attn_t {
	int hidden_size;
	int num_layers;
	int encoder_output_units;

	bool has_proj;
	linear_t *encoder_hidden_proj;
	linear_t *encoder_cell_proj;

	dropout_t **dropouts;
	lstm_cell_t **rnns;

	attention_t *attention;
	dropout_t *dropout_out;

	struct tstack *tstack;
};


static void lstm_attn_init(lstm_attn_t *lstm_attn,
		int input_size, int hidden_size, int num_layers,
		double rate, bool bidirectional, int max_len, int padding_idx)
{
	lstm_attn->hidden_size = hidden_size;
	lstm_attn->num_layers = num_layers;
	lstm_attn->encoder_output_units = hidden_size;


	lstm_attn->has_proj = lstm_attn->encoder_output_units != lstm_attn->hidden_size;
	if (lstm_attn->has_proj) {
		lstm_attn->encoder_hidden_proj = linear_create(lstm_attn->encoder_output_units, hidden_size, true);
		lstm_attn->encoder_cell_proj = linear_create(lstm_attn->encoder_output_units, hidden_size, true);
	}

	lstm_attn->rnns = (lstm_cell_t**) malloc(sizeof(lstm_cell_t*) * num_layers);
	lstm_attn->dropouts = (dropout_t**) malloc(sizeof(dropout_t*) * num_layers);
	for (int i = 0; i < num_layers; i++) {
		lstm_attn->rnns[i] = lstm_cell_create(i == 0 ? input_size + hidden_size : hidden_size, hidden_size);
		lstm_attn->dropouts[i] = dropout_create(rate);
	}


	lstm_attn->attention = attention_create(hidden_size, hidden_size, hidden_size, false);

	lstm_attn->dropout_out = dropout_create(rate);

	lstm_attn->tstack = tstack_create(8);
}

static void lstm_attn_release(lstm_attn_t *lstm_attn)
{
}

lstm_attn_t* lstm_attn_create(int input_size, int hidden_size, int num_layers,
		double rate, bool bidirectional, int max_len, int padding_idx)
{
	lstm_attn_t *lstm_attn = malloc(sizeof(lstm_attn_t));
	lstm_attn_init(lstm_attn, input_size, hidden_size, num_layers, rate, bidirectional, max_len, padding_idx);
	return lstm_attn;
}

void lstm_attn_free(lstm_attn_t *lstm_attn)
{
	lstm_attn_release(lstm_attn);
	free(lstm_attn);
}

tensor_t* lstm_attn_forward(lstm_attn_t *lstm_attn,
		tensor_t *x,
		tensor_t *encoder_outs,
		tensor_t *encoder_hiddens,
		tensor_t *encoder_cells,
		tensor_t *encoder_padding_mask)
{
	// x: tgt_len x batch_size x hidden_size
	// encoder_outs: src_len x batch_size x hidden_size
	// encoder_hiddens: num_layers x batch_size x hidden_size
	// encoder_cells: num_layers x batch_size x hidden_size
	// encoder_padding_mask: seqlen x batch_size x hidden_size
	//
	// out: seqlen x batch_size x hidden_size

	int seq_len = tensor_size(x, 0);
	int batch_size = tensor_size(x, 1);
	int hidden_size = lstm_attn->hidden_size;
	int num_layers = lstm_attn->num_layers;

	tensor_t *outs[seq_len];
	tensor_t *prev_hiddens[seq_len];
	tensor_t *prev_cells[seq_len];

	START_TIMER_ {
	if (lstm_attn->has_proj) {
		encoder_hiddens = linear_forward(lstm_attn->encoder_hidden_proj, encoder_hiddens);
		encoder_cells = linear_forward(lstm_attn->encoder_cell_proj, encoder_cells);
	}

	for (int i = 0; i < num_layers; i++) {
		prev_hiddens[i] = tensor_slice(encoder_hiddens, i, 0);
		prev_cells[i] = tensor_slice(encoder_cells, i, 0);
		tstack_push(lstm_attn->tstack, prev_hiddens[i]);
		tstack_push(lstm_attn->tstack, prev_cells[i]);
	}
	} STOP_TIMER_ ("attn_forward_init")

	int fsizes[] = { batch_size, hidden_size };
	tensor_t *input_feed = tensor_zeros(fsizes, 2);
	START_TIMER_ {
		tstack_push(lstm_attn->tstack, input_feed);
		for (int j = 0; j < seq_len; j++) {
			tensor_t *input_slice;
			tensor_t *input;
			START_TIMER_ {
				input_slice = tensor_slice(x, j, 0);
				input = tensor_concat(input_slice, input_feed, 1);
			} STOP_TIMER_ ("attn_forward_slice_cat")
			tstack_push(lstm_attn->tstack, input_slice);
			tstack_push(lstm_attn->tstack, input);

			tensor_t *hidden = NULL;
			tensor_t *cell = NULL;
			//-> hidden : batch_size x hidden_size
			//-> cell : batch_size x hidden_size
			for (int i = 0; i < num_layers; i++) {
				tensor_pair_t pair = lstm_cell_forward(
						lstm_attn->rnns[i],
						input,
						prev_hiddens[i],
						prev_cells[i]);

				hidden = pair.first;
				cell = pair.second;

				prev_hiddens[i] = hidden;
				prev_cells[i] = cell;

				if (i < num_layers -1)
					input = dropout_forward(lstm_attn->dropout_out, hidden);
			}

			// batch_size x hidden_size
			tensor_t *attn_out;
			tensor_t *dropout_out;
			START_TIMER_ {
				attn_out =  attention_forward(lstm_attn->attention, hidden, encoder_outs, encoder_padding_mask);
				dropout_out = dropout_forward(lstm_attn->dropout_out, attn_out);
			} STOP_TIMER_ ("attn_forward_attn")
			input_feed = dropout_out;
			outs[j] = dropout_out;
		}
	} STOP_TIMER_ ("attn_forward_loop")


	// (seq_len * batch_size) x hidden_size -> useq_len x batch_size x hidden_size
	tensor_t *out;
	START_TIMER_ {
		out = tensor_concat_all(outs, seq_len, 0);
		int view[] = { seq_len, batch_size, lstm_attn->hidden_size };
		tensor_view(out, view, 3);
	} STOP_TIMER_ ("attn_forward_left")

	tstack_push(lstm_attn->tstack, out);
	return out;
}

tensor_t* lstm_attn_inference(lstm_attn_t *lstm_attn,
		tensor_t *x,
		tensor_t *encoder_outs,
		tensor_t *encoder_hiddens,
		tensor_t *encoder_cells,
		tensor_t *encoder_padding_mask)
{
	// x: seqlen x batch_size x hidden_size
	// encoder_out: seqlen x batch_size x hidden_size
	// encoder_hiddens: num_layers x batch_size x hidden_size
	// encoder_cells: num_layers x batch_size x hidden_size
	// encoder_padding_mask: seqlen x batch_size x hidden_size
	//
	// out: seqlen x batch_size x hidden_size

	int seq_len = tensor_size(x, 0);
	int batch_size = tensor_size(x, 1);
	int hidden_size = lstm_attn->hidden_size;
	int num_layers = lstm_attn->num_layers;

	tensor_t *outs[seq_len];
	tensor_t *prev_hiddens[seq_len];
	tensor_t *prev_cells[seq_len];

	for (int i = 0; i < num_layers; i++) {
		if (lstm_attn->has_proj) {
			encoder_hiddens = linear_inference(lstm_attn->encoder_hidden_proj, encoder_hiddens);
			encoder_cells = linear_inference(lstm_attn->encoder_cell_proj, encoder_cells);
		}
		prev_hiddens[i] = tensor_slice(encoder_hiddens, i, 0);
		prev_cells[i] = tensor_slice(encoder_cells, i, 0);
		tstack_push(lstm_attn->tstack, prev_hiddens[i]);
		tstack_push(lstm_attn->tstack, prev_cells[i]);
	}

	int fsizes[] = { batch_size, hidden_size };
	tensor_t *input_feed = tensor_zeros(fsizes, 2);
	tstack_push(lstm_attn->tstack, input_feed);
	for (int j = 0; j < seq_len; j++) {
		tensor_t *input_slice = tensor_slice(x, j, 0);
		tensor_t *input = tensor_concat(input_slice, input_feed, 1);
		tstack_push(lstm_attn->tstack, input_slice);
		tstack_push(lstm_attn->tstack, input);

		tensor_t *hidden = NULL;
		tensor_t *cell = NULL;
		//-> hidden : batch_size x hidden_size
		//-> cell : batch_size x hidden_size
		for (int i = 0; i < num_layers; i++) {
			tensor_pair_t pair = lstm_cell_inference(
					lstm_attn->rnns[i],
					input,
					prev_hiddens[i],
					prev_cells[i]);

			hidden = pair.first;
			cell = pair.second;

			prev_hiddens[i] = hidden;
			prev_cells[i] = cell;

			if (i < num_layers -1)
				input = dropout_inference(lstm_attn->dropout_out, hidden);
		}

		// batch_size x hidden_size
		tensor_t *attn_out =  attention_inference(lstm_attn->attention, hidden, encoder_outs, encoder_padding_mask);
		tensor_t *dropout_out = dropout_inference(lstm_attn->dropout_out, attn_out);
		input_feed = dropout_out;
		outs[j] = dropout_out;
	}


	// seq_len * batch_size x hidden_size -> useq_len x batch_size x hidden_size
	tensor_t *out = tensor_concat_all(outs, seq_len, 0);
	int view[] = { seq_len, batch_size, lstm_attn->hidden_size };
	tensor_view(out, view, 3);

	tstack_push(lstm_attn->tstack, out);
	return out;
}

inline static tensor_t *try_pointwise_add(lstm_attn_t *lstm_attn, tensor_t *x, tensor_t *y)
{
	if (x == NULL)
		return y;
	tensor_t *t = tensor_pointwise_add(x, y);
	tstack_push(lstm_attn->tstack, t);
	return t;
}

tensor_quadruple_t lstm_attn_backward(lstm_attn_t *lstm_attn, tensor_t *dy)
{
	// dy: seq_len x batch_size x hidden_size
	int seq_len = tensor_size(dy, 0);
	int batch_size = tensor_size(dy, 1);
	int hidden_size = lstm_attn->hidden_size;
	int num_layers = lstm_attn->num_layers;

	tensor_t *dy_inputs[seq_len];             // batch_size x hidden_size
	tensor_t *dy_outs[seq_len];             // batch_size x hidden_size
	tensor_t *dy_prev_hiddens[num_layers];
	tensor_t *dy_prev_cells[num_layers];

	int xsizes[] = { seq_len, batch_size, hidden_size };
	int hsizes[] = { num_layers, batch_size, hidden_size };

	tensor_t *dx = tensor_create(xsizes, 3);
	tensor_t *dy_encoder_hiddens = tensor_create(hsizes, 3);
	tensor_t *dy_encoder_cells = tensor_create(hsizes, 3);

	for (int i = 0; i < num_layers; i++) {
		dy_prev_hiddens[i] = tensor_slice(dy_encoder_hiddens, i, 0);
		dy_prev_cells[i] = tensor_slice(dy_encoder_cells, i, 0);
		tstack_push(lstm_attn->tstack, dy_prev_hiddens[i]);
		tstack_push(lstm_attn->tstack, dy_prev_cells[i]);
	}

	for (int j = 0 ; j < seq_len; j++) {
		dy_inputs[j] = tensor_slice(dx, j, 0);
		dy_outs[j] = tensor_slice(dy, j, 0);

		tstack_push(lstm_attn->tstack, dy_inputs[j]);
		tstack_push(lstm_attn->tstack, dy_outs[j]);
	}

	tensor_t *dy_encoder_outs = NULL; //-> seq_len x batch_size x hidden_size
	for (int j = seq_len-1; j >= 0; j--) {
		tensor_t *dy_attn = dropout_backward(lstm_attn->dropout_out, dy_outs[j]);
		tensor_pair_t dy_pair = attention_backward(lstm_attn->attention, dy_attn);
		tensor_t *dy_hidden = dy_pair.first;              // batch_size x hidden_size
		tensor_t *dy_hids =  dy_pair.second;                 // src_len x batch_size x hidden_size
		dy_encoder_outs = try_pointwise_add(lstm_attn, dy_encoder_outs, dy_hids);

		tensor_t *dy_input = NULL;
		tensor_t *dy_cell = NULL;
		for (int i = num_layers - 1; i >= 0; i--) {
			if (i < num_layers - 1) {
				dy_hidden = dropout_backward(lstm_attn->dropout_out, dy_input);
			}
			tensor_triple_t dy_lstm_cell = lstm_cell_backward(lstm_attn->rnns[i], dy_hidden, dy_cell);
			dy_input = dy_lstm_cell.first;
			tensor_t *dy_prev_hidden = dy_lstm_cell.second;
			tensor_t *dy_prev_cell = dy_lstm_cell.third;
			dy_prev_hiddens[i] = try_pointwise_add(lstm_attn, dy_prev_hiddens[i], dy_prev_hidden);
			dy_prev_cells[i] = try_pointwise_add(lstm_attn, dy_prev_cells[i], dy_prev_cell);
		}
		int fsizes[] = { batch_size, hidden_size };
		tensor_pair_t pair = tensor_split(dy_input, fsizes, 1);
		tensor_t *dy_input_slice = pair.first;
		tensor_t *dy_input_feed = pair.second;


		tensor_copy(dy_input_slice, dy_inputs[j]);
		//dy_inputs[j] = dy_input_slice;

		tstack_push(lstm_attn->tstack, dy_input_slice);
		tstack_push(lstm_attn->tstack, dy_input_feed);
	}

	//dx = tensor_concat_all(dy_inputs, seq_len, 0);
	//dy_encoder_hiddens = tensor_concat_all(dy_prev_hiddens, num_layers, 0);
	//dy_encoder_cells = tensor_concat_all(dy_prev_cells, num_layers, 0);

	//	tensor_view(dx, xsizes, 3);
	//	tensor_view(dy_encoder_hiddens, hsizes, 3);
	//	tensor_view(dy_encoder_cells, hsizes, 3);

	tstack_push(lstm_attn->tstack, dx);
	tstack_push(lstm_attn->tstack, dy_encoder_hiddens);
	tstack_push(lstm_attn->tstack, dy_encoder_cells);

	if (lstm_attn->has_proj) {
		dy_encoder_hiddens = linear_backward(lstm_attn->encoder_hidden_proj,
				dy_encoder_hiddens);
		dy_encoder_cells = linear_backward(lstm_attn->encoder_cell_proj,
				dy_encoder_cells);
	}

	tensor_quadruple_t quad = {
		.first = dx,
		.second = dy_encoder_outs,
		.third = dy_encoder_hiddens,
		.fourth = dy_encoder_cells,
	};
	return quad;
}


size_t lstm_attn_params(lstm_attn_t *lstm_attn)
{
	size_t sum = 0;
	if (lstm_attn->has_proj) {
		sum += linear_params(lstm_attn->encoder_hidden_proj);
		sum += linear_params(lstm_attn->encoder_cell_proj);
	}

	for (int i = 0; i < lstm_attn->num_layers; i++) {
		sum += dropout_params(lstm_attn->dropouts[i]);
		sum += lstm_cell_params(lstm_attn->rnns[i]);
	}


	sum += attention_params(lstm_attn->attention);
	sum += dropout_params(lstm_attn->dropout_out);
	return sum;
}

size_t lstm_attn_param_init(lstm_attn_t *lstm_attn, float *param, float *dparam)
{
	size_t offset = 0;
	if (lstm_attn->has_proj) {
		offset += linear_init_params(lstm_attn->encoder_hidden_proj, param+offset, dparam+offset);
		offset += linear_init_params(lstm_attn->encoder_cell_proj, param+offset, dparam+offset);
	}

	for (int i = 0; i < lstm_attn->num_layers; i++) {
		offset += dropout_init_params(lstm_attn->dropouts[i], param+offset, dparam+offset);
		offset += lstm_cell_param_init(lstm_attn->rnns[i], param+offset, dparam+offset);
	}

	offset += attention_param_init(lstm_attn->attention, param+offset, dparam+offset);
	offset += dropout_init_params(lstm_attn->dropout_out, param+offset, dparam+offset);
	return offset;
}


void lstm_attn_clear(lstm_attn_t *lstm_attn)
{
	if (lstm_attn->has_proj) {
		linear_clear(lstm_attn->encoder_hidden_proj);
		linear_clear(lstm_attn->encoder_cell_proj);
	}

	for (int i = 0; i < lstm_attn->num_layers; i++) {
		dropout_clear(lstm_attn->dropouts[i]);
		lstm_cell_clear(lstm_attn->rnns[i]);
	}
	attention_clear(lstm_attn->attention);
	dropout_clear(lstm_attn->dropout_out);
	tstack_clear(lstm_attn->tstack);
}


void lstm_attn_zerograd(lstm_attn_t *lstm_attn)
{
	if (lstm_attn->has_proj) {
		linear_zerograd(lstm_attn->encoder_hidden_proj);
		linear_zerograd(lstm_attn->encoder_cell_proj);
	}

	for (int i = 0; i < lstm_attn->num_layers; i++) {
		dropout_zerograd(lstm_attn->dropouts[i]);
		lstm_cell_zerograd(lstm_attn->rnns[i]);
	}

	attention_zerograd(lstm_attn->attention);
	dropout_zerograd(lstm_attn->dropout_out);
}

void lstm_attn_update(lstm_attn_t *lstm_attn, int N)
{
	if (lstm_attn->has_proj) {
		linear_update(lstm_attn->encoder_hidden_proj, N);
		linear_update(lstm_attn->encoder_cell_proj, N);
	}

	for (int i = 0; i < lstm_attn->num_layers; i++) {
		dropout_update(lstm_attn->dropouts[i], N);
		lstm_cell_update(lstm_attn->rnns[i], N);
	}

	attention_update(lstm_attn->attention, N);
	dropout_update(lstm_attn->dropout_out, N);
}
