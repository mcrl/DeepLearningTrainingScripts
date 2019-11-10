#include <stdbool.h>

#include "options.h"

void parse_opts(struct options *options, int argc, char *argv[], dataset_t *dataset)
{
	// { train 
	options->train.max_epoch = 1;
	options->train.dropout = 0.1;
	options->train.max_toks = 1200;
	// }

	// { encoder
	//     { embedding 
	options->encoder.embedding.freeze_embed = false;
	options->encoder.embedding.embed_dim = 1000;
	//     }
	//     { lstm 
	options->encoder.lstm.embed_dim = options->encoder.embedding.embed_dim;
	options->encoder.lstm.hidden_size = options->encoder.embedding.embed_dim;
	options->encoder.lstm.bidirectional = false;
	options->encoder.lstm.layers = 4;
	//     } 
	//     { dropout 
	options->encoder.dropout.in = options->train.dropout;
	options->encoder.dropout.out = 0;
	//     } 
	// }

	// { decoder
	//     { embedding 
	options->decoder.embedding.freeze_embed = false;
	options->decoder.embedding.embed_dim = 1000;
	options->decoder.encoder_output_units = 512;
	//     }
	//     { lstm 
	options->decoder.lstm.embed_dim = options->decoder.embedding.embed_dim;
	options->decoder.lstm.hidden_size = options->decoder.embedding.embed_dim;
	options->decoder.lstm.bidirectional = false;
	options->decoder.lstm.layers = 4;
	//     } 
	//     { attention 
	options->decoder.attention = true;
	options->decoder.out_embed_dim = 1000;
	//     } 
	//     { dropout 
	options->decoder.dropout.in = options->train.dropout;
	options->decoder.dropout.out = 0;
	//     } 
	// }

	// { share
	options->share.decoder_input_output_embed = false;
	options->share.all_embeddings = false;
	// }

	options->encoder.embedding.num_embeddings = dataset_input_len(dataset);
	options->encoder.embedding.padding_idx= dataset_input_padding_idx(dataset);
	options->encoder.lstm.max_len = dataset_dict_max_input_len(dataset);
	options->decoder.embedding.num_embeddings = dataset_target_len(dataset);
	options->decoder.embedding.padding_idx = dataset_target_padding_idx(dataset);
	options->decoder.lstm.max_len = dataset_dict_max_input_len(dataset);
}
