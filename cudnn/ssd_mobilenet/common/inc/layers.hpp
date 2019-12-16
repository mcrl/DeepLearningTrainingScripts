#ifndef _CUDNN_LAYERS_HPP_
#define _CUDNN_LAYERS_HPP_

#include <cudnn.h>
#include <string>
#include <cfloat>
#include "util.hpp"
#include "parser.hpp"

class PaddingInfo {
	public:
		PaddingInfo(int pad_h = 0, int pad_w = 0, int stride_h = 0, int stride_w = 0) {
			this->pad_h = pad_h;
			this->pad_w = pad_w;
			this->stride_h = stride_h;
			this->stride_w = stride_w;
			this->dilation_h = 1;
			this->dilation_w = 1;
		}
		int pad_h;		/* zero-padding height */
		int pad_w;		/* zero-padding width */
		int stride_h;
		int stride_w;
		int dilation_h;
		int dilation_w;
};

class TensorInfo {
	public:
		TensorInfo() {
			n = 0;
			c = 0;
			h = 0;
			w = 0;
			format = CUDNN_TENSOR_NCHW;
			dataType = CUDNN_DATA_FLOAT;
		};
		TensorInfo(int n, int c, int h, int w, 
				cudnnTensorFormat_t format=CUDNN_TENSOR_NCHW, 
				cudnnDataType_t dataType=CUDNN_DATA_FLOAT) : 
			n(n), c(c), h(h), w(w), format(format), dataType(dataType) { };
		int n;
		int c;
		int h;
		int w;
		cudnnTensorFormat_t format;
		cudnnDataType_t dataType;
};

class Tensor {
	public:
		Tensor() {};
		Tensor(const TensorInfo info) {
			this->info = info;
		};
		void* getAddress() { return this->data; };
		TensorInfo getInfo() { return this->info; };
	private:
		void *data;
		TensorInfo info;
};

class Layer {
	public:
		Layer() {
			//cout << "Layer Constructor" << endl; 
			one = 1.0f;
			zero = 0.0f;
		};
		virtual ~Layer() { 
			//cout << "Layer Deconstructor" << endl; 
		};
		virtual float* forward(float* input) = 0;
		virtual float* func(float* input, float *input2) { 
		   return nullptr; };
		virtual void setup(void) = 0;
		virtual TensorInfo getOutputInfo(void) = 0;
		virtual string getLayerName(void) {
			return layerName;
		};
	protected:
		float one;
		float zero;
		string layerName;
		cudnnHandle_t cudnnHandle;
};

class ConvolutionLayer : public Layer {
	public:
		ConvolutionLayer(const cudnnHandle_t &cudnnHandle,
				const TensorInfo inputInfo,
				const TensorInfo filterInfo,
				const PaddingInfo padding,
				const string weightPath,
				const string biasPath,
				const int groupNum = 1,
				const string &layerName="ConvolutionLayer",
				const TensorInfo outputInfoHint=TensorInfo());
		~ConvolutionLayer();
		void setup(void);
		float* forward(float *input);
		TensorInfo getOutputInfo(void) { return outputInfo; };
	protected:
		cudnnTensorDescriptor_t inputDesc;
		cudnnTensorDescriptor_t afterConvDesc;
		cudnnTensorDescriptor_t convBiasDesc;
		cudnnConvolutionDescriptor_t convDesc;
		cudnnFilterDescriptor_t filterDesc;
		cudnnConvolutionFwdAlgo_t fwdAlgo;
		bool useFwdAlgoProfile;
		int fwdAlgoCountReq;
		int fwdAlgoCountRet;
		size_t convWorkspaceBytes;
		int groupNum;
		PaddingInfo paddingInfo;
		TensorInfo inputInfo;
		TensorInfo outputInfo;
		TensorInfo filterInfo;
		float *filter;
		float *bias;
		float *afterConv;
		void *convWorkspace;
		string weightPath;
		string biasPath;
		int dilation_h;
		int dilation_w;
	private:
		void _getConvFwdAlgo(void);
};

class FullyConnectedLayer : public Layer {
	public:
		FullyConnectedLayer(const cudnnHandle_t &cudnnHandle,
				const TensorInfo &inputInfo,
				const TensorInfo &filterInfo,
				const string weightPath,
				const string biasPath,
				const string &layerName="FullyConnectedLayer"); 
		~FullyConnectedLayer();
		void setup(void);
		float* forward(float *input);
		TensorInfo getOutputInfo(void) { return outputInfo; };
	private:
		cudnnTensorDescriptor_t inputDesc;
		cudnnConvolutionDescriptor_t fcnDesc;
		cudnnFilterDescriptor_t fcnFilterDesc;
		cudnnTensorDescriptor_t afterFcnDesc;
		cudnnTensorDescriptor_t fcnBiasDesc;
		cudnnConvolutionFwdAlgo_t fcnAlgo;

		TensorInfo inputInfo;
		TensorInfo outputInfo;
		TensorInfo filterInfo;
		size_t fcnWorkspaceBytes;
		string weightPath;
		string biasPath;
		float *filter;
		float *bias;
		float *afterFcn;
		void *fcnWorkspace;
};

class IdentityLayer : public Layer {
	public:
		IdentityLayer(const cudnnHandle_t &cudnnHandle,
				const TensorInfo &inputInfo,
				const string &layerName="IdentityLayer"); 
		~IdentityLayer();
		void setup(void);
		float* forward(float *input);
		TensorInfo getOutputInfo(void) { return outputInfo; };

	private:
		cudnnTensorDescriptor_t inputDesc;
		TensorInfo inputInfo;
		TensorInfo outputInfo;
};

class AddLayer : public Layer {
	public:
		AddLayer(const cudnnHandle_t &cudnnHandle,
				const TensorInfo &inputInfo,
				const TensorInfo &inputInfo2,
				const string &layerName="AddLayer");
		~AddLayer();
		void setup(void);
		float* forward(float *input);
		float* func(float* input, float *input2);
		TensorInfo getOutputInfo(void) { return outputInfo; };

	private:
		cudnnTensorDescriptor_t inputDesc;
		TensorInfo inputInfo;
		TensorInfo inputInfo2;
		TensorInfo outputInfo;
};

class SoftmaxLayer : public Layer {
	public:
		SoftmaxLayer(const cudnnHandle_t &cudnnHandle,
				const TensorInfo &inputInfo,
				const string &layerName="SoftmaxLayer"); 
		~SoftmaxLayer();
		void setup(void);
		float* forward(float *input);
		TensorInfo getOutputInfo(void) { return outputInfo; };

	private:
		TensorInfo inputInfo;
		TensorInfo outputInfo;
		cudnnTensorDescriptor_t inputDesc;
		float *afterSoftmax;
};

class PoolingLayer : public Layer {
	public:
		PoolingLayer(const cudnnHandle_t &cudnnHandle,
				const TensorInfo &inputInfo,
				const TensorInfo &poolingInfo,
				const PaddingInfo &paddingInfo,
				const cudnnPoolingMode_t poolMode=CUDNN_POOLING_MAX,
				const string &layerName="PoolingLayer");
		~PoolingLayer();
		void setup(void);
		float* forward(float *input);
		TensorInfo getOutputInfo(void) { return outputInfo; };

	private:
		TensorInfo inputInfo;
		TensorInfo outputInfo;
		TensorInfo poolingInfo;
		PaddingInfo paddingInfo;
		cudnnTensorDescriptor_t inputDesc;
		cudnnPoolingDescriptor_t poolDesc;
		cudnnTensorDescriptor_t afterPoolDesc;
		cudnnPoolingMode_t poolMode;
		float *afterPool;
};

class BatchNormalizationLayer : public Layer {
	public:
		BatchNormalizationLayer(const cudnnHandle_t &cudnnHandle,
				const TensorInfo &inputInfo,
				const string &betaPath,	const string &gammaPath, const string &movingMeanPath, const string &movingVariancePath,
				const double &epslion = 1e-03,
				const string &layerName="BatchNormalizationLayer");

		~BatchNormalizationLayer(); 
		void setup(void);
		float* forward(float *input);
		TensorInfo getOutputInfo(void) { return outputInfo; };

	private:
		cudnnTensorDescriptor_t inputDesc;
		cudnnTensorDescriptor_t bnDesc;
		cudnnTensorDescriptor_t afterBnDesc;
		TensorInfo inputInfo;
		TensorInfo outputInfo;
		double epslion;
		string betaPath;
		string gammaPath;
		string movingMeanPath;
		string movingVariancePath;
		float* bnScale;
		float* bnBias;
		float* bnResultRunningMean;
		float* bnResultRunningVar;
		float* afterBn;
		bool isTraining;
};

class ReluLayer : public Layer {
	public:
		ReluLayer( const cudnnHandle_t &cudnnHandle,
			const TensorInfo &inputInfo,
			const double &bound = 20.0,
			const string &layerName="ReluLayer");
		~ReluLayer();
		void setup(void);
		float* forward(float *input);
		TensorInfo getOutputInfo(void) { return outputInfo; };

	private:
		cudnnTensorDescriptor_t inputDesc;
		cudnnTensorDescriptor_t afterActDesc;
		cudnnActivationDescriptor_t actDesc;
		TensorInfo inputInfo;
		TensorInfo outputInfo;
		double bound;
		float* afterAct;
};

class PReluLayer : public Layer {
	public:
		PReluLayer( const cudnnHandle_t &cudnnHandle,
			const TensorInfo &inputInfo,
			const string weightPath,
			const string &layerName="PReluLayer");
		~PReluLayer();
		void setup(void);
		float* forward(float *input);
		TensorInfo getOutputInfo(void) { return outputInfo; };

	private:
		TensorInfo inputInfo;
		TensorInfo outputInfo;
		string weightPath;
		float* weight;
		float* afterAct;
};

class SigmoidLayer : public Layer {
	public:
		SigmoidLayer( const cudnnHandle_t &cudnnHandle,
			const TensorInfo &inputInfo,
			const string &layerName="SigmoidLayer");
		~SigmoidLayer();
		void setup(void);
		float* forward(float *input);
		TensorInfo getOutputInfo(void) { return outputInfo; };

	private:
		cudnnTensorDescriptor_t inputDesc;
		cudnnTensorDescriptor_t afterActDesc;
		cudnnActivationDescriptor_t actDesc;
		TensorInfo inputInfo;
		TensorInfo outputInfo;
		float* afterAct;
};

class ResidualLayer : public Layer {
	public:
		ResidualLayer(const cudnnHandle_t &cudnnHandle,
				const TensorInfo &inputInfo,
				const TensorInfo &outputInfo,
				vector<Layer*> &layers,
				Layer *leftLayer,
				Layer *funcLayer,
				Layer *reluLayer,
				const string &layerName="ResidualLayer");
		~ResidualLayer();
		void setup(void);
		float* forward(float *input);
		TensorInfo getOutputInfo(void) { return outputInfo; };

	private:
		TensorInfo inputInfo;
		TensorInfo outputInfo;
		vector<Layer*> layers;
		Layer *leftLayer;
		Layer *funcLayer;
		Layer *reluLayer;
};

class ConcatenationLayer : public Layer {
	public:
		ConcatenationLayer(const cudnnHandle_t &cudnnHandle,
				const TensorInfo &inputInfo,
				const TensorInfo &inputInfo2,
				const string &layerName="ConcatenationLayer"); 
		~ConcatenationLayer();
		void setup(void);
		float* func(float *input, float *input2); 
		float* forward(float* input) { return input; };
		TensorInfo getOutputInfo(void) { return outputInfo; };
	private:
		TensorInfo inputInfo;
		TensorInfo inputInfo2;
		TensorInfo outputInfo;
		float* afterConcat;
};

class ChannelShuffleLayer : public Layer {
	public:
		ChannelShuffleLayer(const cudnnHandle_t &cudnnHandle,
				const TensorInfo &inputInfo,
				const int groupNum,
				const string &layerName="ChannelShuffleLayer");
		~ChannelShuffleLayer();
		void setup(void);
		float* forward(float* input);
		TensorInfo getOutputInfo(void) { return outputInfo; };

	private:
		TensorInfo inputInfo;
		TensorInfo outputInfo;
		int groupNum;
		int channelsPerGroup;
		float *afterShuffle;
};

typedef struct {
		int additionalPadding;
		int totalPadding;
} OddPaddingInfo;

class ConvolutionTFStyleLayer : public ConvolutionLayer {
	public:
		ConvolutionTFStyleLayer(
				const cudnnHandle_t &cudnnHandle,
				const TensorInfo inputInfo,
				const TensorInfo filterInfo,
				const PaddingInfo padding,
				const string weightPath,
				const string biasPath,
				const int groupNum = 1,
				const string &layerName="ConvolutionTFStyleLayer");
		~ConvolutionTFStyleLayer();
		void setup(void);
		float* forward(float* input); 		
	protected:
	private:
		OddPaddingInfo _computePadding(int inputSize, int filterSize, int stride, int dilation);
		float *paddedInput;
		TensorInfo inputInfoBackup;
		OddPaddingInfo rowsOdd;
		OddPaddingInfo colsOdd;

};

class DepthwiseConvLayer : public ConvolutionLayer {
	public:
		DepthwiseConvLayer(
				const cudnnHandle_t &cudnnHandle,
				const TensorInfo inputInfo,
				const TensorInfo filterInfo,
				const PaddingInfo padding,
				const string weightPath,
				const string biasPath,
				const int groupNum = 1,
				const string &layerName="DepthwiseConvLayer");
		~DepthwiseConvLayer();
		void setup(void);
		float* forward(float* input); 		
	protected:
	private:

};
#endif
