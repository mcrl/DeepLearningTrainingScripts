#ifndef _LSTM_H_
#define _LSTM_H_

#include "tensor.h"
#include "layer.h"
#include "dropout.h"

#include "layer/embedding.h"
#include <vector>
#include <cudnn.h>

#include "kernels.h"

using namespace std;


class LSTM{
public:
	LSTM(int input_size, int num_layers, int  hidden_size, float dropout_out){
		_input_size = input_size;
		_num_layers = num_layers;
		_hidden_size = hidden_size;
		_dropout_out = dropout_out;
		
			
		chkCUDNN(cudnnCreate(&cudnn));
	}
	~LSTM(){

	}

	cudnnHandle_t cudnn;
	cudnnTensorDescriptor_t *x_desc;
	cudnnTensorDescriptor_t *y_desc;
    cudnnTensorDescriptor_t hx_desc, cx_desc, hy_desc, cy_desc;
	cudnnDropoutDescriptor_t dropoutDesc;
	cudnnRNNDescriptor_t rnnDesc;
	cudnnRNNMode_t RNNMode;
	cudnnRNNAlgo_t algo;
	cudnnFilterDescriptor_t wDesc;

	size_t stateSize;
	void *states;
	void *workspace;
    void *reserveSpace;
	void *weight;


	int _seqLen;


	float _dropout_out;
	int _input_size;
	int _num_layers;
	int _hidden_size;

	size_t _weight_size;

	
    size_t workSize;
    size_t reserveSize;

	
	void Init();
	void Prepare(int seqLen, int batch_size);

	//void Forward(std::vector<Tensor *> input, std::vector<Tensor*> output);
	void Forward(Tensor *x, 
				Tensor *cx, 
				Tensor* hx, 
				Tensor *y, 
				Tensor *cy, 
				Tensor *hy);
};



#endif /* _LSTM_H_ */

