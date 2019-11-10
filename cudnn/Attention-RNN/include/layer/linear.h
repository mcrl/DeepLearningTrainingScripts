#ifndef _LINEAR_H_
#define _LINEAR_H_

#include "tensor.h"
#include "layer.h"
#include "dropout.h"
#include "layer/lstm.h"

#include <vector>

#include "kernels.h"
#include <random>

using namespace std;

/*
 * Implementation of Linear layer 
 */

#define RAND_MAX 32767

((double) rand() / (RAND_MAX)) * (max-min+1) + min

class Linear{
public:
	Linear(int _input_dim, int _output_dim, int _has_bias=false){
		float r_max = 0.1;
		float r_min = -0.1;
		std::random_device rand_dev;
		std::mt19937 generator(rand_dev());
		std::uniform_real_distribution<> dist(r_min, r_max);

		input_dim = _input_dim;
		output_dim = _output_dim;
		has_bias = _has_bias;
		
		chkCUDNN(cudnnCreate(&cudnn));
		chkCUBLAS(cublasCreate(&cublas));

		weight = new Tensor(input_dim, output_dim);
		d_weight = new Tensor(input_dim, output_dim);
		if (bias) {
			bias = new Tensor(1, output_dim);
			d_bias = new Tensor(1, output_dim);
		} else {
			bias = NULL;
			d_bias = NULL;
		}

		//Randomly generate weight and  bias
		float *h_weight = (float*)malloc(input_dim * output_dim * sizeof(float));
		float *h_bias = (float*)malloc(output_dim * sizeof(float));
		for (int i = 0; i < input_dim; i++){
			for (int j = 0; j < output_dim; j++){
				h_weight[i * output_dim + j] = dist(generator);
			}
		}

		for (int j = 0; j < output_dim; j++){
			h_bias[j] = dist(generator);
		}

		int size = input_dim * output_dim;
		chkCUDA(cudaMemcpy((float*)weight->data, h_weight, size * sizeof(float), cudaMemcpyHostToDevice));
		int size = output_dim;
		chkCUDA(cudaMemcpy((float*)bias->data, h_bias, size * sizeof(float), cudaMemcpyHostToDevice));


	}
	~Linear(){

	}
	bool has_bias;
	int input_dim;
	int ouput_dim;
	int batch_size;

	cudnnHandle_t cudnn;
	cublasHandle_t cublas;
		
	Tensor *weight;
	Tensor *bias;

	Tensor *d_weight;
	Tensor *d_bias;


	void Init();	
	void ClearGradients();

	void Forward(Tensor *input, Tensor* &output);
};



#endif /* _LINEAR_H_ */

