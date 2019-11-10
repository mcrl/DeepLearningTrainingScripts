#ifndef __PASRE_OPTS_H__
#define __PASRE_OPTS_H__

#include "dataset.h"

#include <stdbool.h>

struct embedding_options {
	int num_embeddings;
	int embed_dim;
	int padding_idx;
	bool freeze_embed;
};

struct dropout_options {
	double in;
	double out;
};

struct lstm_options {
	int embed_dim;
	int hidden_size;
	int layers;
	bool bidirectional;
	int max_len;
};

struct encoder_options {
	struct embedding_options embedding;
	struct lstm_options lstm;
	struct dropout_options dropout;
};

struct decoder_options {
	struct embedding_options embedding;
	struct lstm_options lstm;
	struct dropout_options dropout;
	int out_embed_dim;
	bool attention;
	int encoder_output_units;
}; 

struct options {
	struct {
		int max_epoch;
		double dropout;
		int max_toks;
	} train;

	struct encoder_options encoder;
	struct decoder_options decoder;

	struct {
		bool decoder_input_output_embed;
		bool all_embeddings;
	} share;

};

void parse_opts(struct options *options, int argc, char *argv[], dataset_t *dataset);

#endif //__PASRE_OPTS_H__
