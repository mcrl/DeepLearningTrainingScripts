
#include "layers.hpp"
#include "kernel.h"

/*
 * ConvolutionLayer
 */
ConvolutionLayer::ConvolutionLayer(
		const cudnnHandle_t &cudnnHandle,
		const TensorInfo inputInfo,
		const TensorInfo filterInfo,
		const PaddingInfo padding,
		const string weightPath,
		const string biasPath,
		const int groupNum,
		const string &layerName,
		const TensorInfo outputInfoHint) {
	this->cudnnHandle = cudnnHandle;
	this->inputInfo = inputInfo;
	this->filterInfo = filterInfo;
	this->paddingInfo = padding;
	this->dilation_h = 1;
	this->dilation_w = 1;
	this->groupNum = groupNum;
	this->weightPath = weightPath;
	this->biasPath= biasPath;
	this->outputInfo = outputInfoHint;
	this->layerName = layerName;
	this->bias = NULL;
	this->useFwdAlgoProfile = true;
	setup();
};

ConvolutionLayer::~ConvolutionLayer() {
	checkCUDNN(cudnnDestroyTensorDescriptor(inputDesc));
	checkCUDNN(cudnnDestroyTensorDescriptor(afterConvDesc));
	checkCUDNN(cudnnDestroyConvolutionDescriptor(convDesc));
	checkCUDNN(cudnnDestroyFilterDescriptor(filterDesc));
	checkCUDA(cudaFree(filter));
	checkCUDA(cudaFree(afterConv));
	checkCUDA(cudaFree(convWorkspace));
	if(biasPath.compare("None") != 0) {
		checkCUDNN(cudnnDestroyTensorDescriptor(convBiasDesc));
		checkCUDA(cudaFree(bias));
	}
};

void ConvolutionLayer::_getConvFwdAlgo(void) {
	if(useFwdAlgoProfile) {
		int fwdAlgoCountReq;
		int fwdAlgoCountRet;
		checkCUDNN(cudnnGetConvolutionForwardAlgorithmMaxCount(cudnnHandle, &fwdAlgoCountReq));
		cudnnConvolutionFwdAlgoPerf_t *fwdAlgoPerfResult = new cudnnConvolutionFwdAlgoPerf_t[fwdAlgoCountReq];
		checkCUDNN(cudnnFindConvolutionForwardAlgorithm(cudnnHandle,
					inputDesc, filterDesc, convDesc, afterConvDesc,
					fwdAlgoCountReq, &fwdAlgoCountRet, fwdAlgoPerfResult));
		float fwdAlgoMinTime = 1e9;
		for(int i = 0; i < fwdAlgoCountRet; i++) {
			if(fwdAlgoPerfResult[i].status == CUDNN_STATUS_SUCCESS) {
				if(fwdAlgoPerfResult[i].time < fwdAlgoMinTime) {
					fwdAlgoMinTime = fwdAlgoPerfResult[i].time;
					fwdAlgo = fwdAlgoPerfResult[i].algo;
				}
			}
		}
		delete[] fwdAlgoPerfResult;
	} else {
		checkCUDNN(cudnnGetConvolutionForwardAlgorithm(cudnnHandle,
					inputDesc, filterDesc, convDesc, afterConvDesc,
					CUDNN_CONVOLUTION_FWD_PREFER_FASTEST, 0, &fwdAlgo));
	}
}

void ConvolutionLayer::setup() {
	checkCUDNN(cudnnCreateTensorDescriptor(&inputDesc));
	checkCUDNN(cudnnCreateTensorDescriptor(&afterConvDesc));
	checkCUDNN(cudnnCreateConvolutionDescriptor(&convDesc));
	checkCUDNN(cudnnCreateFilterDescriptor(&filterDesc));

	int inputChannel = filterInfo.c / groupNum;
	checkCUDNN(cudnnSetFilter4dDescriptor(filterDesc,
				filterInfo.dataType, filterInfo.format,
				filterInfo.n, inputChannel, filterInfo.h, filterInfo.w));
	checkCUDNN(cudnnSetConvolution2dDescriptor(convDesc,
				paddingInfo.pad_h, paddingInfo.pad_w, paddingInfo.stride_h, paddingInfo.stride_w,
				dilation_h, dilation_w, CUDNN_CROSS_CORRELATION, filterInfo.dataType));
	checkCUDNN(cudnnSetConvolutionGroupCount(convDesc, groupNum));

	outputInfo.n = inputInfo.n;
	outputInfo.c = filterInfo.n;
	outputInfo.h = CALC_SIZE(inputInfo.h, filterInfo.h, paddingInfo.pad_h, paddingInfo.stride_h);
	outputInfo.w = CALC_SIZE(inputInfo.w, filterInfo.w, paddingInfo.pad_w, paddingInfo.stride_w);

	MALLOC_TENSOR_FLOAT(&afterConv, outputInfo.n, outputInfo.c, outputInfo.h, outputInfo.w);
	checkCUDNN(cudnnSetTensor4dDescriptor(inputDesc,
				inputInfo.format, inputInfo.dataType, 
				inputInfo.n, inputInfo.c, inputInfo.h, inputInfo.w));
	checkCUDNN(cudnnSetTensor4dDescriptor(afterConvDesc,
				outputInfo.format, outputInfo.dataType,
				outputInfo.n, outputInfo.c, outputInfo.h, outputInfo.w));
	
	_getConvFwdAlgo();
	checkCUDNN(cudnnGetConvolutionForwardWorkspaceSize(cudnnHandle,
				inputDesc, filterDesc, convDesc, afterConvDesc,
				fwdAlgo, &convWorkspaceBytes));
	checkCUDA(cudaMalloc(&convWorkspace, convWorkspaceBytes));

	/* Data load */
	MALLOC_TENSOR_FLOAT(&filter, filterInfo.n, inputChannel, filterInfo.h, filterInfo.w);
	NPYParser parser;
	parser.load(weightPath);
	parser.parse();
	int temp_out = filterInfo.n * (inputChannel) * filterInfo.h * filterInfo.w;
	checkCUDA(cudaMemcpy(filter, parser.getDataStartAddr(), sizeof(float) * temp_out, cudaMemcpyHostToDevice));

	if(biasPath.compare("None") != 0) {
		checkCUDNN(cudnnCreateTensorDescriptor(&convBiasDesc));
		checkCUDNN(cudnnSetTensor4dDescriptor(convBiasDesc, outputInfo.format, outputInfo.dataType, 1, filterInfo.n, 1, 1));
		MALLOC_TENSOR_FLOAT(&bias, filterInfo.n,1, 1, 1);
		parser.load(biasPath);
		parser.parse();
		checkCUDA(cudaMemcpy(bias, parser.getDataStartAddr(), sizeof(float) * filterInfo.n, cudaMemcpyHostToDevice));
	}
};

float* ConvolutionLayer::forward(float* input) {
	checkCUDNN(cudnnConvolutionForward(cudnnHandle, &one,
				inputDesc, input, filterDesc, filter, convDesc,
				fwdAlgo, convWorkspace, convWorkspaceBytes,
				&zero, afterConvDesc, afterConv));
	if(biasPath.compare("None") != 0)
		checkCUDNN(cudnnAddTensor(cudnnHandle, &one, convBiasDesc, bias, &one, afterConvDesc, afterConv));
	return afterConv;
};

/*
 * FullyConnectedLayer
 */

FullyConnectedLayer::FullyConnectedLayer(const cudnnHandle_t &cudnnHandle,
		const TensorInfo &inputInfo,
		const TensorInfo &filterInfo,
		const string weightPath,
		const string biasPath,
		const string &layerName) {
	this->cudnnHandle = cudnnHandle;
	this->inputInfo = inputInfo;
	this->filterInfo = filterInfo;
	this->weightPath = weightPath;
	this->biasPath= biasPath;
	this->outputInfo= inputInfo;
	this->layerName = layerName;
	setup();
};

FullyConnectedLayer::~FullyConnectedLayer() {
	checkCUDNN(cudnnDestroyTensorDescriptor(inputDesc));
	checkCUDNN(cudnnDestroyTensorDescriptor(afterFcnDesc));
	checkCUDNN(cudnnDestroyTensorDescriptor(fcnBiasDesc));
	checkCUDNN(cudnnDestroyFilterDescriptor(fcnFilterDesc));
	checkCUDA(cudaFree(filter));
	checkCUDA(cudaFree(bias));
	checkCUDA(cudaFree(afterFcn));
	checkCUDA(cudaFree(fcnWorkspace));
};

void FullyConnectedLayer::setup() {
	outputInfo.c = filterInfo.n;
	checkCUDNN(cudnnCreateTensorDescriptor(&inputDesc));
	checkCUDNN(cudnnSetTensor4dDescriptor(inputDesc,
				inputInfo.format, inputInfo.dataType, inputInfo.n, inputInfo.c, inputInfo.h, inputInfo.w));
	checkCUDNN(cudnnCreateConvolutionDescriptor(&fcnDesc));
	checkCUDNN(cudnnCreateTensorDescriptor(&fcnBiasDesc));
	checkCUDNN(cudnnCreateTensorDescriptor(&afterFcnDesc));
	checkCUDNN(cudnnCreateFilterDescriptor(&fcnFilterDesc));
	checkCUDNN(cudnnSetConvolution2dDescriptor(fcnDesc, 0, 0, 1, 1, 1, 1, CUDNN_CROSS_CORRELATION, inputInfo.dataType));
	checkCUDNN(cudnnSetFilter4dDescriptor(fcnFilterDesc, filterInfo.dataType, filterInfo.format, filterInfo.n, filterInfo.c, filterInfo.h, filterInfo.w));
	checkCUDNN(cudnnSetTensor4dDescriptor(afterFcnDesc, outputInfo.format, outputInfo.dataType,  outputInfo.n, outputInfo.c, outputInfo.h, outputInfo.w));
	checkCUDNN(cudnnSetTensor4dDescriptor(fcnBiasDesc, outputInfo.format, outputInfo.dataType, 1, outputInfo.c, 1, 1));

	MALLOC_TENSOR_FLOAT(&filter, filterInfo.n, filterInfo.c, filterInfo.h, filterInfo.w);
	MALLOC_TENSOR_FLOAT(&afterFcn, outputInfo.n, outputInfo.c, outputInfo.h, outputInfo.w);
	MALLOC_TENSOR_FLOAT(&bias, 1, outputInfo.c, 1, 1);
	NPYParser parser;
	parser.load(weightPath);
	parser.parse();
	int temp_out = filterInfo.n * filterInfo.c * filterInfo.h * filterInfo.w;
	checkCUDA(cudaMemcpy(filter, parser.getDataStartAddr(), sizeof(float) * temp_out, cudaMemcpyHostToDevice));
	parser.load(biasPath);
	parser.parse();
	checkCUDA(cudaMemcpy(bias, parser.getDataStartAddr(), sizeof(float) * outputInfo.c, cudaMemcpyHostToDevice));

	checkCUDNN(cudnnGetConvolutionForwardAlgorithm(cudnnHandle,
				inputDesc, fcnFilterDesc, fcnDesc, afterFcnDesc,
				CUDNN_CONVOLUTION_FWD_PREFER_FASTEST, 0, &fcnAlgo));
	checkCUDNN(cudnnGetConvolutionForwardWorkspaceSize(cudnnHandle,
				inputDesc, fcnFilterDesc, fcnDesc, afterFcnDesc,
				fcnAlgo, &fcnWorkspaceBytes));
	checkCUDA(cudaMalloc(&fcnWorkspace, fcnWorkspaceBytes));
};

float* FullyConnectedLayer::forward(float* input) {
	checkCUDNN(cudnnConvolutionForward(cudnnHandle, &one,
				inputDesc, input, fcnFilterDesc, filter, fcnDesc,
				fcnAlgo, fcnWorkspace, fcnWorkspaceBytes,
				&zero, afterFcnDesc, afterFcn));
	checkCUDNN(cudnnAddTensor(cudnnHandle, &one, fcnBiasDesc, bias, &one, afterFcnDesc, afterFcn));
	//DUMP_TENSOR(afterFcn, "./dump/" + layerName + ".dump", outputInfo.n * outputInfo.c * outputInfo.h * outputInfo.w);
	return afterFcn;
};

IdentityLayer::IdentityLayer(const cudnnHandle_t &cudnnHandle,
		const TensorInfo &inputInfo,
		const string &layerName) {
	this->cudnnHandle = cudnnHandle;
	this->inputInfo = inputInfo;
	this->outputInfo= inputInfo;
	this->layerName = layerName;
}

IdentityLayer::~IdentityLayer() {
};

void IdentityLayer::setup(void) {
};

float* IdentityLayer::forward(float* input) {
	return input;
};

AddLayer::AddLayer(const cudnnHandle_t &cudnnHandle,
		const TensorInfo &inputInfo,
		const TensorInfo &inputInfo2,
		const string &layerName) {
	this->cudnnHandle = cudnnHandle;
	this->inputInfo = inputInfo;
	this->inputInfo2 = inputInfo2;
	this->outputInfo= inputInfo;
	this->layerName = layerName;
	setup();
};

AddLayer::~AddLayer() {
	checkCUDNN(cudnnDestroyTensorDescriptor(inputDesc));
};

void AddLayer::setup(void) {
	checkCUDNN(cudnnCreateTensorDescriptor(&inputDesc));
	checkCUDNN(cudnnSetTensor4dDescriptor(inputDesc,
				inputInfo.format, inputInfo.dataType, inputInfo.n, inputInfo.c, inputInfo.h, inputInfo.w));
};

float* AddLayer::func(float *input, float *input2) {
	checkCUDNN(cudnnAddTensor(cudnnHandle, &one, inputDesc, input, &one, inputDesc, input2));
	return input2;
};

float* AddLayer::forward(float* input) {
	return input;
};

SoftmaxLayer::SoftmaxLayer(const cudnnHandle_t &cudnnHandle,
		const TensorInfo &inputInfo,
		const string &layerName) {
	this->cudnnHandle = cudnnHandle;
	this->inputInfo = inputInfo;
	this->outputInfo= inputInfo;
	this->layerName = layerName;
	setup();
};

SoftmaxLayer::~SoftmaxLayer() {
	checkCUDNN(cudnnDestroyTensorDescriptor(inputDesc));
	checkCUDA(cudaFree(afterSoftmax));
};

void SoftmaxLayer::setup(void) {
	checkCUDNN(cudnnCreateTensorDescriptor(&inputDesc));
	checkCUDNN(cudnnSetTensor4dDescriptor(inputDesc,
				inputInfo.format, inputInfo.dataType, inputInfo.n, inputInfo.c, inputInfo.h, inputInfo.w));
	MALLOC_TENSOR_FLOAT(&afterSoftmax, outputInfo.n, outputInfo.c, outputInfo.h, outputInfo.w);
}

float* SoftmaxLayer::forward(float* input) {
	checkCUDNN(cudnnSoftmaxForward(cudnnHandle, CUDNN_SOFTMAX_ACCURATE, CUDNN_SOFTMAX_MODE_CHANNEL, &one, inputDesc, input, &zero, inputDesc, afterSoftmax));
	return afterSoftmax;
};

PoolingLayer::PoolingLayer(
		const cudnnHandle_t &cudnnHandle,
		const TensorInfo &inputInfo,
		const TensorInfo &poolingInfo,
		const PaddingInfo &paddingInfo,
		const cudnnPoolingMode_t poolMode,
		const string &layerName) {
	this->cudnnHandle = cudnnHandle;
	this->inputInfo = inputInfo;
	this->poolingInfo= poolingInfo;
	this->outputInfo = inputInfo;
	this->paddingInfo = paddingInfo;
	this->layerName = layerName;
	this->poolMode = poolMode;

	setup();
};

PoolingLayer::~PoolingLayer() {
	checkCUDNN(cudnnDestroyTensorDescriptor(inputDesc));
	checkCUDNN(cudnnDestroyTensorDescriptor(afterPoolDesc));
	checkCUDNN(cudnnDestroyPoolingDescriptor(poolDesc));
	checkCUDA(cudaFree(afterPool));
};

void PoolingLayer::setup(void) {
	checkCUDNN(cudnnCreateTensorDescriptor(&inputDesc));
	checkCUDNN(cudnnSetTensor4dDescriptor(inputDesc,
				inputInfo.format, inputInfo.dataType, inputInfo.n, inputInfo.c, inputInfo.h, inputInfo.w));
	checkCUDNN(cudnnCreatePoolingDescriptor(&poolDesc));
	checkCUDNN(cudnnSetPooling2dDescriptor(poolDesc,
				poolMode,
				CUDNN_NOT_PROPAGATE_NAN,
				poolingInfo.h, poolingInfo.w,
				paddingInfo.pad_h, paddingInfo.pad_w, paddingInfo.stride_h, paddingInfo.stride_w));
	checkCUDNN(cudnnCreateTensorDescriptor(&afterPoolDesc));

	outputInfo.h = CALC_SIZE(inputInfo.h, poolingInfo.h, paddingInfo.pad_h, paddingInfo.stride_h);
	outputInfo.w = CALC_SIZE(inputInfo.w, poolingInfo.w, paddingInfo.pad_w, paddingInfo.stride_w);

	checkCUDNN(cudnnSetTensor4dDescriptor(afterPoolDesc,
				outputInfo.format, outputInfo.dataType, outputInfo.n, outputInfo.c, outputInfo.h, outputInfo.w));

	MALLOC_TENSOR_FLOAT(&afterPool, outputInfo.n, outputInfo.c, outputInfo.h, outputInfo.w);
}

float* PoolingLayer::forward(float* input) {
	checkCUDNN(cudnnPoolingForward(cudnnHandle, poolDesc,
				&one, inputDesc, input,
				&zero,afterPoolDesc, afterPool));
	return afterPool;
};

BatchNormalizationLayer::BatchNormalizationLayer(
		const cudnnHandle_t &cudnnHandle,
		const TensorInfo &inputInfo,
		const string &betaPath,	const string &gammaPath, const string &movingMeanPath, const string &movingVariancePath,
		const double &epslion,
		const string &layerName) {
	this->cudnnHandle = cudnnHandle;
	this->inputInfo = inputInfo;
	this->betaPath= betaPath;
	this->gammaPath= gammaPath;
	this->movingMeanPath = movingMeanPath;
	this->movingVariancePath= movingVariancePath;
	this->outputInfo = inputInfo;
	this->epslion = epslion;
	this->isTraining = false;
	this->layerName = layerName;
	setup();
};

BatchNormalizationLayer::~BatchNormalizationLayer() {
	checkCUDNN(cudnnDestroyTensorDescriptor(inputDesc));
	checkCUDNN(cudnnDestroyTensorDescriptor(bnDesc));
	checkCUDNN(cudnnDestroyTensorDescriptor(afterBnDesc));
	checkCUDA(cudaFree(bnScale));
	checkCUDA(cudaFree(bnBias));
	checkCUDA(cudaFree(bnResultRunningVar));
	checkCUDA(cudaFree(bnResultRunningMean));
	checkCUDA(cudaFree(afterBn));
};

void BatchNormalizationLayer::setup(void) {
	checkCUDNN(cudnnCreateTensorDescriptor(&bnDesc));
	checkCUDNN(cudnnCreateTensorDescriptor(&afterBnDesc));
	checkCUDNN(cudnnCreateTensorDescriptor(&inputDesc));
	checkCUDNN(cudnnSetTensor4dDescriptor(inputDesc,
				inputInfo.format, inputInfo.dataType, inputInfo.n, inputInfo.c, inputInfo.h, inputInfo.w));
	checkCUDNN(cudnnDeriveBNTensorDescriptor(bnDesc, inputDesc, CUDNN_BATCHNORM_SPATIAL));
	checkCUDNN(cudnnSetTensor4dDescriptor(afterBnDesc,
				inputInfo.format, inputInfo.dataType, inputInfo.n, inputInfo.c,
				inputInfo.h, inputInfo.w));
	MALLOC_TENSOR_FLOAT(&bnScale, 1, inputInfo.c, 1, 1);
	MALLOC_TENSOR_FLOAT(&bnBias, 1, inputInfo.c, 1, 1);
	MALLOC_TENSOR_FLOAT(&bnResultRunningMean, 1, inputInfo.c, 1, 1);
	MALLOC_TENSOR_FLOAT(&bnResultRunningVar, 1, inputInfo.c, 1, 1);
	MALLOC_TENSOR_FLOAT(&afterBn, inputInfo.n, inputInfo.c, inputInfo.h, inputInfo.w);

	NPYParser parser;
	parser.load(gammaPath);
	parser.parse();
	checkCUDA(cudaMemcpy(bnScale, parser.getDataStartAddr(), sizeof(float) * inputInfo.c, cudaMemcpyHostToDevice));
	parser.load(betaPath);
	parser.parse();
	checkCUDA(cudaMemcpy(bnBias, parser.getDataStartAddr(), sizeof(float) * inputInfo.c, cudaMemcpyHostToDevice));
	parser.load(movingMeanPath);
	parser.parse();
	checkCUDA(cudaMemcpy(bnResultRunningMean, parser.getDataStartAddr(), sizeof(float) * inputInfo.c, cudaMemcpyHostToDevice));
	parser.load(movingVariancePath);
	parser.parse();
	checkCUDA(cudaMemcpy(bnResultRunningVar, parser.getDataStartAddr(), sizeof(float) * inputInfo.c, cudaMemcpyHostToDevice));
};

float* BatchNormalizationLayer::forward(float* input) {
	checkCUDNN(cudnnBatchNormalizationForwardInference(cudnnHandle,
				CUDNN_BATCHNORM_SPATIAL, &one, &zero,
				inputDesc, input,
				afterBnDesc, afterBn, bnDesc,
				bnScale, bnBias,
				bnResultRunningMean, bnResultRunningVar,
				epslion));
	return afterBn;
};

ReluLayer::ReluLayer( const cudnnHandle_t &cudnnHandle,
		const TensorInfo &inputInfo,
		const double &bound,
		const string &layerName) {
	this->cudnnHandle = cudnnHandle;
	this->bound = bound;
	this->inputInfo = inputInfo;
	this->outputInfo = inputInfo;
	this->layerName = layerName;
	setup();
};

ReluLayer::~ReluLayer() {
	checkCUDNN(cudnnDestroyTensorDescriptor(inputDesc));
	checkCUDNN(cudnnDestroyTensorDescriptor(afterActDesc));
	checkCUDNN(cudnnDestroyActivationDescriptor(actDesc));
	checkCUDA(cudaFree(afterAct));
};

void ReluLayer::setup(void) {
	checkCUDNN(cudnnCreateTensorDescriptor(&afterActDesc));
	checkCUDNN(cudnnCreateActivationDescriptor(&actDesc));
	checkCUDNN(cudnnCreateTensorDescriptor(&inputDesc));
	checkCUDNN(cudnnSetTensor4dDescriptor(inputDesc,
				inputInfo.format, inputInfo.dataType, inputInfo.n, inputInfo.c, inputInfo.h, inputInfo.w));
	cudnnActivationMode_t activationMode = CUDNN_ACTIVATION_CLIPPED_RELU;
	if(bound == DBL_MAX) {
		activationMode = CUDNN_ACTIVATION_RELU;
	}
	checkCUDNN(cudnnSetActivationDescriptor(actDesc,
				activationMode, CUDNN_NOT_PROPAGATE_NAN, bound));
	checkCUDNN(cudnnSetTensor4dDescriptor(afterActDesc,
				inputInfo.format, inputInfo.dataType, inputInfo.n, inputInfo.c, inputInfo.h, inputInfo.w));
	MALLOC_TENSOR_FLOAT(&afterAct, inputInfo.n, inputInfo.c, inputInfo.h, inputInfo.w);
}

float* ReluLayer::forward(float* input) {
	checkCUDNN(cudnnActivationForward(cudnnHandle, actDesc,
				&one, inputDesc, input, &zero,
				afterActDesc, afterAct));
	return afterAct;
};

/*
 * Implementation of PReluLayer class 
 */

PReluLayer::PReluLayer(const cudnnHandle_t &cudnnHandle,
			const TensorInfo &inputInfo,
			const string weightPath,
			const string &layerName) {
	this->cudnnHandle = cudnnHandle;
	this->inputInfo = inputInfo;
	this->outputInfo = inputInfo;
	this->weightPath = weightPath;
	this->layerName = layerName;
	setup();
};

PReluLayer::~PReluLayer() {
	checkCUDA(cudaFree(weight));
	checkCUDA(cudaFree(afterAct));
};

void PReluLayer::setup(void) {
	MALLOC_TENSOR_FLOAT(&afterAct, inputInfo.n, inputInfo.c, inputInfo.h, inputInfo.w);
	MALLOC_TENSOR_FLOAT(&weight, 1, inputInfo.c, 1, 1);
	NPYParser parser;
	parser.load(weightPath);
	parser.parse();
	checkCUDA(cudaMemcpy(weight, parser.getDataStartAddr(), sizeof(float) * inputInfo.c, cudaMemcpyHostToDevice));
};

float* PReluLayer::forward(float* input) {
	cuda_prelu(input, afterAct, weight, inputInfo.n, inputInfo.c, inputInfo.h * inputInfo.w);
	return afterAct;
};

/*
 * Implementation of SigmoidLayer class 
 */
SigmoidLayer::SigmoidLayer( const cudnnHandle_t &cudnnHandle,
		const TensorInfo &inputInfo,
		const string &layerName) {
	this->cudnnHandle = cudnnHandle;
	this->inputInfo = inputInfo;
	this->outputInfo = inputInfo;
	this->layerName = layerName;
	setup();
};

SigmoidLayer::~SigmoidLayer() {
	checkCUDNN(cudnnDestroyTensorDescriptor(inputDesc));
	checkCUDNN(cudnnDestroyTensorDescriptor(afterActDesc));
	checkCUDNN(cudnnDestroyActivationDescriptor(actDesc));
	checkCUDA(cudaFree(afterAct));
};

void SigmoidLayer::setup(void) {
	checkCUDNN(cudnnCreateTensorDescriptor(&afterActDesc));
	checkCUDNN(cudnnCreateActivationDescriptor(&actDesc));
	checkCUDNN(cudnnCreateTensorDescriptor(&inputDesc));
	checkCUDNN(cudnnSetTensor4dDescriptor(inputDesc,
				inputInfo.format, inputInfo.dataType, inputInfo.n, inputInfo.c, inputInfo.h, inputInfo.w));
	cudnnActivationMode_t activationMode = CUDNN_ACTIVATION_SIGMOID;
	checkCUDNN(cudnnSetActivationDescriptor(actDesc,
				activationMode, CUDNN_NOT_PROPAGATE_NAN, 0));
	checkCUDNN(cudnnSetTensor4dDescriptor(afterActDesc,
				inputInfo.format, inputInfo.dataType, inputInfo.n, inputInfo.c, inputInfo.h, inputInfo.w));
	MALLOC_TENSOR_FLOAT(&afterAct, inputInfo.n, inputInfo.c, inputInfo.h, inputInfo.w);
}

float* SigmoidLayer::forward(float* input) {
	checkCUDNN(cudnnActivationForward(cudnnHandle, actDesc,
				&one, inputDesc, input, &zero,
				afterActDesc, afterAct));
	return afterAct;
};

ResidualLayer::ResidualLayer(const cudnnHandle_t &cudnnHandle,
				const TensorInfo &inputInfo,
				const TensorInfo &outputInfo,
				vector<Layer*> &layers,
				Layer *leftLayer,
				Layer *funcLayer,
				Layer *reluLayer,
				const string &layerName) {
	this->cudnnHandle = cudnnHandle;
	this->inputInfo = inputInfo;
	this->outputInfo = outputInfo;
	this->layers = layers;
	this->leftLayer = leftLayer;
	this->funcLayer = funcLayer;
	this->reluLayer= reluLayer;
	this->layerName = layerName;
	setup();
}

ResidualLayer::~ResidualLayer() {
	for(Layer *layer : layers) {
		delete layer;
	}
	delete leftLayer;
	delete funcLayer;
	delete reluLayer;
};

void ResidualLayer::setup(void) {};

float* ResidualLayer::forward(float *input) {
	float *residual;
	float *output;
	residual = leftLayer->forward(input);
	for(Layer *layer : layers) {
		output = layer->forward(input);
		input = output;
	}
	output = funcLayer->func(residual, output);
	output = reluLayer->forward(output);
	return output;
};

ConcatenationLayer::ConcatenationLayer(const cudnnHandle_t &cudnnHandle,
				const TensorInfo &inputInfo,
				const TensorInfo &inputInfo2,
				const string &layerName) {
	this->cudnnHandle = cudnnHandle;
	this->inputInfo = inputInfo;
	this->inputInfo2 = inputInfo2;
	this->outputInfo= inputInfo;
	this->layerName = layerName;
	setup();
};

ConcatenationLayer::~ConcatenationLayer() {
	checkCUDA(cudaFree(afterConcat));
};

void ConcatenationLayer::setup(void) {
	outputInfo.c = inputInfo.c + inputInfo2.c;
	MALLOC_TENSOR_FLOAT(&afterConcat, outputInfo.n, outputInfo.c, outputInfo.h, outputInfo.w);
};

float* ConcatenationLayer:: func(float *input, float *input2) {
	cuda_concatenate(input, input2, afterConcat, inputInfo.n, inputInfo.c, inputInfo2.c, inputInfo.h * inputInfo.w);
	return afterConcat;
};

ChannelShuffleLayer::ChannelShuffleLayer(const cudnnHandle_t &cudnnHandle,
				const TensorInfo &inputInfo,
				const int groupNum,
				const string &layerName) {
	this->cudnnHandle = cudnnHandle;
	this->inputInfo = inputInfo;
	this->outputInfo= inputInfo;
	this->groupNum = groupNum;
	this->layerName = layerName;
	setup();
};

ChannelShuffleLayer::~ChannelShuffleLayer() {
	checkCUDA(cudaFree(afterShuffle));
};

void ChannelShuffleLayer::setup(void) {
	channelsPerGroup = inputInfo.c / groupNum;
	MALLOC_TENSOR_FLOAT(&afterShuffle, outputInfo.n, outputInfo.c, outputInfo.h, outputInfo.w);
};

float* ChannelShuffleLayer::forward(float* input) {
	cuda_shuffle(input, afterShuffle, inputInfo.n, groupNum, channelsPerGroup, inputInfo.h * inputInfo.w);
	return afterShuffle;
};

DepthwiseConvLayer::DepthwiseConvLayer(const cudnnHandle_t &cudnnHandle,
				const TensorInfo inputInfo,
				const TensorInfo filterInfo,
				const PaddingInfo padding,
				const string weightPath,
				const string biasPath,
				const int groupNum,
				const string &layerName) : ConvolutionLayer(cudnnHandle, inputInfo, filterInfo,
						padding, weightPath, biasPath, groupNum, layerName) {
	/* Unnecessary memory allocation */
};

DepthwiseConvLayer::~DepthwiseConvLayer() {
};

void DepthwiseConvLayer::setup(void) {
	ConvolutionLayer::setup();
};

float* DepthwiseConvLayer::forward(float *input) {
	cuda_depthwise_conv(input, afterConv, filter, bias, outputInfo.c, inputInfo.h, inputInfo.w, outputInfo.h, outputInfo.w, paddingInfo.stride_h, 0, filterInfo.h);
	return afterConv;
};

ConvolutionTFStyleLayer::ConvolutionTFStyleLayer(const cudnnHandle_t &cudnnHandle,
				const TensorInfo inputInfo,
				const TensorInfo filterInfo,
				const PaddingInfo padding,
				const string weightPath,
				const string biasPath,
				const int groupNum,
				const string &layerName) : ConvolutionLayer(cudnnHandle, inputInfo, filterInfo,
						padding, weightPath, biasPath, groupNum, layerName) {
	paddedInput = nullptr;
	setup();
};

ConvolutionTFStyleLayer::~ConvolutionTFStyleLayer() {
	if(rowsOdd.additionalPadding || colsOdd.additionalPadding) {
		checkCUDA(cudaFree(paddedInput));
	}
};

void ConvolutionTFStyleLayer::setup(void) {
	/* Check Odd padding */
	rowsOdd = _computePadding(inputInfo.h, filterInfo.h, paddingInfo.stride_h, dilation_h);
	colsOdd = _computePadding(inputInfo.w, filterInfo.w, paddingInfo.stride_w, dilation_w);
	if(rowsOdd.additionalPadding || colsOdd.additionalPadding) {
		inputInfoBackup = inputInfo;
		inputInfo.h += rowsOdd.additionalPadding;
		inputInfo.w += colsOdd.additionalPadding;
		paddingInfo.pad_h -= rowsOdd.totalPadding;
		paddingInfo.pad_w -= colsOdd.totalPadding;
		MALLOC_TENSOR_FLOAT(&paddedInput, inputInfo.n, inputInfo.c, inputInfo.h, inputInfo.w);
	}
	ConvolutionLayer::setup();
};

float* ConvolutionTFStyleLayer::forward(float *input) {
	float *srcInput = input;
	int offset = 0;
	if(rowsOdd.additionalPadding || colsOdd.additionalPadding) {
		cuda_pad(input, paddedInput, inputInfo.n, inputInfo.c, inputInfoBackup.h, inputInfoBackup.w, inputInfo.h, inputInfo.w);
		srcInput = paddedInput;
		offset = 1;
	}

	if(groupNum == 1) {
		checkCUDNN(cudnnConvolutionForward(cudnnHandle, &one,
			inputDesc, srcInput, filterDesc, filter, convDesc,
			fwdAlgo, convWorkspace, convWorkspaceBytes,
			&zero, afterConvDesc, afterConv));
		if(biasPath.compare("None") != 0)
			checkCUDNN(cudnnAddTensor(cudnnHandle, &one, convBiasDesc, bias, &one, afterConvDesc, afterConv));
	} else {
		cuda_depthwise_conv(srcInput, afterConv, filter, bias, outputInfo.c, inputInfo.h, inputInfo.w, outputInfo.h, outputInfo.w, paddingInfo.stride_h, offset, filterInfo.h);
	}
	return afterConv;
};

OddPaddingInfo ConvolutionTFStyleLayer::_computePadding(int inputSize, int filterSize, int stride, int dilation) 
{
	OddPaddingInfo odd;
	int effectiveFilterSize = (filterSize - 1) * dilation + 1;
	int outSize = (inputSize + stride - 1) / stride;
	odd.totalPadding = max(0, (outSize - 1) * stride + effectiveFilterSize - inputSize);
	odd.additionalPadding = (odd.totalPadding % 2 != 0);
	return odd;
};

