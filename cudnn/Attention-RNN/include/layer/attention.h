#ifndef _ATTENTION_LSTM_H_
#define _ATTENTION_LSTM_H_

#include "tensor.h"
#include "layer.h"
#include "dropout.h"
#include "layer/lstm.h"

#include "layer/linear.h"
#include <vector>

#include "kernels.h"

using namespace std;

/*
 * Implementation of TransformerEncoder at fairseq
 */

class Attention{
public:
	Attention(int _input_embed_dim, int _source_embed_dim, int _output_embed_dim, bool _bias){
		this->input_embed_dim = _input_embed_dim;
		this->source_embed_dim = _source_embed_dim;
		this->output_embed_dim = _output_embed_dim;
		this->bias = _bias;

		chkCUDNN(cudnnCreate(&cudnn));
	}
	~Attention(){

	}

	Linear* input_proj;
	Linear* output_proj;

	Tensor *x;

	int input_embed_dim;
	int source_embed_dim;
	int output_embed_dim;
	bool bias;

	cudnnHandle_t cudnn;


	void Init();	

	void Forward(Tensor *input, std::vector<Tensor*> &output);
};



#endif /* _ATTENTION_LSTM_H_ */

