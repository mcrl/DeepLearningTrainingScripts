#ifndef _ENCODER_LSTM_H_
#define _ENCODER_LSTM_H_

#include "tensor.h"
#include "layer.h"
#include "dropout.h"
#include "layer/lstm.h"

#include "layer/embedding.h"
#include <vector>

#include "kernels.h"

using namespace std;

/*
 * Implementation of TransformerEncoder at fairseq
 */

class EncoderLSTM{
public:
	EncoderLSTM(){
	}
	~EncoderLSTM(){

	}
	int embed_dim;
	int padding_idx;
	int num_layers;
	int hidden_size;

	embedding_param_t *embed_token_param;
	embedding_t *embed_token;
	Dropout *dropout1;
	Dropout *dropout2;
	LSTM **rnn;



	Tensor *x_embed;
	Tensor *x_dropout;
	Tensor *y_dropout;
	Tensor *x_trans;

	Tensor *final_hiddens;
	Tensor *final_cells;

	void Init();	
	void Do_lstm();

	void Forward(Tensor *input, std::vector<Tensor*> &output);
};



#endif /* _ENCODER_LSTM_H_ */

