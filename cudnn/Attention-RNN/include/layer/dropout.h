#ifndef _DROPOUT_H_
#define _DROPOUT_H_ 

#include "tensor.h"
//#include "layer.h"
#include "utils.h"
#include <cudnn.h>

#include "layer/embedding.h"
#include <vector>

using namespace std;

/*
 * Implementation of TransformerEncoder at fairseq
 */

class Dropout {
public:
	Dropout(float p){
		_rate = p;
	}
	~Dropout(){

	}

	void Init();	

	void Allocate(Tensor *input);
	void Forward(Tensor *input, Tensor* output);
	void Backward(Tensor *input_grad, Tensor* output_grad);
	
	float _rate;
	cudnnHandle_t cudnn;
	cudnnDropoutDescriptor_t desc;
	size_t state_size;
	size_t reserve_size;
	cudnnTensorDescriptor_t in_out_desc;

	void *state, *reserved_space;
};



#endif /* _DROPOUT_H_ */

