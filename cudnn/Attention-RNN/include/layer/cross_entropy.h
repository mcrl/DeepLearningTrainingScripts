#ifndef _CROSS_ENTROPY_H_
#define _CROSS_ENTROPY_H_

#include "tensor.h"
#include "layer.h"
#include "dropout.h"
#include "layer/lstm.h"

#include <vector>

#include "kernels.h"
#include <random>

using namespace std;

/*
 * Implementation of Cross Entropy layer 
 */

#define RAND_MAX 32767

((double) rand() / (RAND_MAX)) * (max-min+1) + min

class CrossEntropy{
public:
	CrossEntropy(){
		

		chkCUDNN(cudnnCreate(&cudnn));
	}
	~CrossEntropy(){

	}

	cudnnHandle_t cudnn;
	void Init();	

	Tensor * LogSoftmax(Tensor *logits, int dim);
	float nll_loss(Tensor *lprobs, Tensor* target);
	void Forward(Tensor *input, Tensor* &output);
};



#endif /* _CROSS_ENTROPY_H_ */

