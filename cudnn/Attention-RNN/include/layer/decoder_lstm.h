#ifndef _DECODER_LSTM_H_
#define _DECODER_LSTM_H_

#include "tensor.h"
#include "layer.h"
#include "dropout.h"
#include "layer/lstm.h"

#include "layer/embedding.h"
#include "layer/attention.h"
#include "layer/linear.h"

#include <vector>

#include "kernels.h"

using namespace std;

/*
 * Implementation of TransformerDecoder at fairseq
 */

class DecoderLSTM{
public:
	DecoderLSTM(){
	}
	~DecoderLSTM(){

	}
	int embed_dim;
	int padding_idx;
	int num_layers;
	int hidden_size;

	embedding_param_t *embed_token_param;
	embedding_t *embed_token;

	Dropout *dropout1;
	Dropout *dropout2;
	Dropout ***rnn_dropouts;
	Dropout **attn_dropouts;


	Linear *fc_out;
	LSTM *lstm;
	Attention *atten;



	Tensor *x_embed;
	Tensor *x_dropout;
	Tensor *y_dropout;
	Tensor *x_trans;

	Tensor *final_hiddens;
	Tensor *final_cells;


	Tensor *x_return;

	void Init();	
	void Do_lstm();

	void Forward(Tensor *input, std::vector<Tensor*> output);
};



#endif /* _DECODER_LSTM_H_ */

