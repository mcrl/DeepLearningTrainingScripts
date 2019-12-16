#include <iostream>
#include <cudnn.h>
#include <vector>
#include <tuple>
#include <string>
#include <cfloat>
#include <unistd.h>
#include <chrono>
#include <iomanip>
#include "layers.hpp"
#include "parser.hpp"
#include "util.hpp"
#include "kernel.h"
#include "nms.hpp"

using namespace std;
using namespace chrono;
const int NUM_PRED = 6;

class SSD_MobileNet_v1_224 {
	public:
		SSD_MobileNet_v1_224(int channel = 3, int height = 300, int width = 300, int numClasses = 91) : channel(channel), height(height), width(width), numClasses(numClasses) {
			checkCUDNN(cudnnCreate(&cudnnHandle));
			dataPath = "./data/model/";
			inputPath = "./data/coco-2017-val/";
			numBoxes = 4;
		};

		~SSD_MobileNet_v1_224() {
			checkCUDNN(cudnnDestroy(cudnnHandle));
			checkCUDA(cudaFree(inputImage));
			checkCUDA(cudaFree(prior));
			checkCUDA(cudaFree(predBoxes));
			delete scoresHost;
			delete boxesHost;

			for(Layer *layer : backboneLayers)delete layer;
			for(Layer *layer : blockLayers)	delete layer;
			for(Layer *layer : classificationLayers) delete layer;
			for(Layer *layer : regressionLayers) delete layer;
			for(Layer *layer : classifiConcatLayers) delete layer;
			for(Layer *layer : regressionConcatLayers) delete layer;
		};

		void init() {
			/* Input */
			checkCUDA(cudaMalloc(&inputImage, channel * height * width * sizeof(float)));

			/* Backbone Layers */
			network.clear();
			backboneLayers.clear();
			Layer *convolutionLayer = new ConvolutionTFStyleLayer(
					cudnnHandle,
					TensorInfo(1,3,height,width),
					TensorInfo(32,3,3,3),
					PaddingInfo(1,1,2,2),
					dataPath + "backbone.0.0.weight.npy",
					dataPath + "backbone.0.0.BatchNorm.bias.npy");
			backboneLayers.push_back(convolutionLayer);

			Layer *reluLayer = new ReluLayer(cudnnHandle, convolutionLayer->getOutputInfo(), 6.0f);
			backboneLayers.push_back(reluLayer);

			int predicIdx = 0;
			TensorInfo predictionInfo[NUM_PRED];
			TensorInfo outputInfo = reluLayer->getOutputInfo();
			outputInfo = addDwpwLayer(1, outputInfo, TensorInfo(32,32,3,3), PaddingInfo(1,1,1,1), TensorInfo(64,32,1,1), PaddingInfo(0,0,1,1));
			outputInfo = addDwpwLayer(2, outputInfo, TensorInfo(64,64,3,3), PaddingInfo(1,1,2,2), TensorInfo(128,64,1,1), PaddingInfo(0,0,1,1));
			outputInfo = addDwpwLayer(3, outputInfo, TensorInfo(128,128,3,3), PaddingInfo(1,1,1,1), TensorInfo(128,128,1,1), PaddingInfo(0,0,1,1));
			outputInfo = addDwpwLayer(4, outputInfo, TensorInfo(128,128,3,3), PaddingInfo(1,1,2,2), TensorInfo(256,128,1,1), PaddingInfo(0,0,1,1));
			outputInfo = addDwpwLayer(5, outputInfo, TensorInfo(256,256,3,3), PaddingInfo(1,1,1,1), TensorInfo(256,256,1,1), PaddingInfo(0,0,1,1));
			outputInfo = addDwpwLayer(6, outputInfo, TensorInfo(256,256,3,3), PaddingInfo(1,1,2,2), TensorInfo(512,256,1,1), PaddingInfo(0,0,1,1));
			for(int i = 7; i < 12; ++i)
				outputInfo = addDwpwLayer(i, outputInfo, TensorInfo(512,512,3,3), PaddingInfo(1,1,1,1), TensorInfo(512,512,1,1), PaddingInfo(0,0,1,1));
			predictionInfo[predicIdx++] = outputInfo; /* it's output goes to the prediction layer */
			outputInfo = addDwpwLayer(12, outputInfo, TensorInfo(512,512,3,3), PaddingInfo(1,1,2,2), TensorInfo(1024,512,1,1), PaddingInfo(0,0,1,1));
			outputInfo = addDwpwLayer(13, outputInfo, TensorInfo(1024,1024,3,3), PaddingInfo(1,1,1,1), TensorInfo(1024,1024,1,1), PaddingInfo(0,0,1,1));
			predictionInfo[predicIdx++] = outputInfo; /* it's output goes to the prediction layer */

			/* Block Layers */
			outputInfo = addBlockLayer(0, outputInfo, 1024, 256, 512);
			predictionInfo[predicIdx++] = outputInfo;
			outputInfo = addBlockLayer(1, outputInfo, 512, 128, 256);
			predictionInfo[predicIdx++] = outputInfo;
			outputInfo = addBlockLayer(2, outputInfo, 256, 128, 256);
			predictionInfo[predicIdx++] = outputInfo;
			outputInfo = addBlockLayer(3, outputInfo, 256, 64, 128);
			predictionInfo[predicIdx++] = outputInfo;
			
			/* Prediction Layers */
			predicIdx = 0;
			TensorInfo classificationOutputInfos[NUM_PRED];
			classificationOutputInfos[predicIdx] = addClassificationLayer(0, predictionInfo[predicIdx], 512, 91, 3); predicIdx++;
			classificationOutputInfos[predicIdx] = addClassificationLayer(1, predictionInfo[predicIdx], 1024, 91, 6);predicIdx++;
			classificationOutputInfos[predicIdx] = addClassificationLayer(2, predictionInfo[predicIdx], 512, 91, 6);predicIdx++;
			classificationOutputInfos[predicIdx] = addClassificationLayer(3, predictionInfo[predicIdx], 256, 91, 6);predicIdx++;
			classificationOutputInfos[predicIdx] = addClassificationLayer(4, predictionInfo[predicIdx], 256, 91, 6);predicIdx++;
			classificationOutputInfos[predicIdx] = addClassificationLayer(5, predictionInfo[predicIdx], 128, 91, 6);

			predicIdx = 0;
			TensorInfo regressionOutputInfos[NUM_PRED];
			regressionOutputInfos[predicIdx] = addRegressionLayer(0, predictionInfo[predicIdx], 512, 91, 3); predicIdx++;
			regressionOutputInfos[predicIdx] = addRegressionLayer(1, predictionInfo[predicIdx], 1024, 91, 6);predicIdx++;
			regressionOutputInfos[predicIdx] = addRegressionLayer(2, predictionInfo[predicIdx], 512, 91, 6);predicIdx++;
			regressionOutputInfos[predicIdx] = addRegressionLayer(3, predictionInfo[predicIdx], 256, 91, 6);predicIdx++;
			regressionOutputInfos[predicIdx] = addRegressionLayer(4, predictionInfo[predicIdx], 256, 91, 6);predicIdx++;
			regressionOutputInfos[predicIdx] = addRegressionLayer(5, predictionInfo[predicIdx], 128, 91, 6);

			reorder(classificationOutputInfos, numClasses);
			reorder(regressionOutputInfos, numBoxes);

			/* Concat outputs */
			Layer *concatLayer0 = new ConcatenationLayer(cudnnHandle, classificationOutputInfos[0], classificationOutputInfos[1], "class.concat_0");
			classifiConcatLayers.push_back(concatLayer0);
			Layer *concatLayer1 = new ConcatenationLayer(cudnnHandle, concatLayer0->getOutputInfo(), classificationOutputInfos[2], "class.concat_1");
			classifiConcatLayers.push_back(concatLayer1);
			Layer *concatLayer2 = new ConcatenationLayer(cudnnHandle, concatLayer1->getOutputInfo(), classificationOutputInfos[3], "class.concat_2");
			classifiConcatLayers.push_back(concatLayer2);
			Layer *concatLayer3 = new ConcatenationLayer(cudnnHandle, concatLayer2->getOutputInfo(), classificationOutputInfos[4], "class.concat_3");
			classifiConcatLayers.push_back(concatLayer3);
			Layer *concatLayer4 = new ConcatenationLayer(cudnnHandle, concatLayer3->getOutputInfo(), classificationOutputInfos[5], "class.concat_4");
			classifiConcatLayers.push_back(concatLayer4);
			outputInfo = concatLayer4->getOutputInfo();
			scoresHost = new float[outputInfo.n * outputInfo.c * outputInfo.h * outputInfo.w];

			concatLayer0 = new ConcatenationLayer(cudnnHandle, regressionOutputInfos[0], regressionOutputInfos[1], "regression.concat_0");
			regressionConcatLayers.push_back(concatLayer0);
			concatLayer1 = new ConcatenationLayer(cudnnHandle, concatLayer0->getOutputInfo(), regressionOutputInfos[2], "regression.concat_1");
			regressionConcatLayers.push_back(concatLayer1);
			concatLayer2 = new ConcatenationLayer(cudnnHandle, concatLayer1->getOutputInfo(), regressionOutputInfos[3], "regression.concat_2");
			regressionConcatLayers.push_back(concatLayer2);
			concatLayer3 = new ConcatenationLayer(cudnnHandle, concatLayer2->getOutputInfo(), regressionOutputInfos[4], "regression.concat_3");
			regressionConcatLayers.push_back(concatLayer3);
			concatLayer4 = new ConcatenationLayer(cudnnHandle, concatLayer3->getOutputInfo(), regressionOutputInfos[5], "regression.concat_4");
			regressionConcatLayers.push_back(concatLayer4);

			/* Filter Boxes */
			outputInfo = concatLayer4->getOutputInfo();
			totalRowSize = outputInfo.c;
			MALLOC_TENSOR_FLOAT(&prior, outputInfo.n, outputInfo.c, outputInfo.h, outputInfo.w);
			MALLOC_TENSOR_FLOAT(&predBoxes, outputInfo.n, outputInfo.c, outputInfo.h, outputInfo.w);
			boxesHost = new float[outputInfo.n * outputInfo.c * outputInfo.h * outputInfo.w];
			NPYParser parser;
			parser.load(dataPath + "priors.npy");
			parser.parse();
			checkCUDA(cudaMemcpy(prior, parser.getDataStartAddr(), sizeof(float) * (outputInfo.c * outputInfo.w), cudaMemcpyHostToDevice));
		};

		void reorder(TensorInfo (&infos)[NUM_PRED], int colSize) {
			for(int i = 0; i < NUM_PRED; ++i) {
				int rowSize = (infos[i].c * infos[i].h * infos[i].w);
				if(rowSize % colSize != 0) {
					cout << "Error " << rowSize << " " << colSize << endl;
					return;
				}
				rowSize /= colSize;
				infos[i].c = rowSize;
				infos[i].h = 1;
				infos[i].w = colSize;
			}
			return;
		};

		TensorInfo addBlockLayer(int layerIndex, TensorInfo inputInfo, int inChannels, int midChannels, int outChannels) {
			string dataPrefix = "extras." + to_string(layerIndex) + ".";
			Layer *convolutionLayer = new ConvolutionLayer(cudnnHandle, inputInfo, TensorInfo(midChannels, inChannels,1,1), PaddingInfo(0,0,1,1),
					dataPath + dataPrefix + "0.weight.npy", 
					dataPath + dataPrefix + "0.bias.npy", 
					1, "blockconv_" + to_string(layerIndex));
			blockLayers.push_back(convolutionLayer);

			Layer *reluLayer = new ReluLayer(cudnnHandle, convolutionLayer->getOutputInfo(), 6.0f,
					"blockrelu_" + to_string(layerIndex));
			blockLayers.push_back(reluLayer);

			Layer *convolutionTFSLayer = new ConvolutionTFStyleLayer(cudnnHandle, reluLayer->getOutputInfo(),TensorInfo(outChannels, midChannels, 3, 3), PaddingInfo(1,1,2,2),
					dataPath + dataPrefix + "2.weight.npy", 
					dataPath + dataPrefix + "2.bias.npy", 
					1, "blockconv2_" + to_string(layerIndex));
			blockLayers.push_back(convolutionTFSLayer);

			Layer *relu2Layer = new ReluLayer(cudnnHandle, convolutionTFSLayer->getOutputInfo(), 6.0f,
					"blockrelu2_" + to_string(layerIndex));
			blockLayers.push_back(relu2Layer);
			return relu2Layer->getOutputInfo();
		};

		TensorInfo addClassificationLayer(int layerIndex, TensorInfo inputInfo, int inChannels, int numClass, int numAnchors) {
			string dataPrefix = "predictors." + to_string(layerIndex) + ".";
			Layer *classificationLayer = new ConvolutionLayer(cudnnHandle, inputInfo, TensorInfo(numClass * numAnchors, inChannels,1,1), PaddingInfo(0,0,1,1),
					dataPath + dataPrefix + "classification.weight.npy", 
					dataPath + dataPrefix + "classification.bias.npy", 
					1, "predict.classify_" + to_string(layerIndex),
					TensorInfo(0,0,0,0, CUDNN_TENSOR_NHWC));
			classificationLayers.push_back(classificationLayer);
			Layer *sigmoidLayer = new SigmoidLayer(cudnnHandle, classificationLayer->getOutputInfo(), "predict.classify.sigmoid_" + to_string(layerIndex));
			classificationLayers.push_back(sigmoidLayer);
			return sigmoidLayer->getOutputInfo();
		};

		TensorInfo addRegressionLayer(int layerIndex, TensorInfo inputInfo, int inChannels, int numClass, int numAnchors) {
			string dataPrefix = "predictors." + to_string(layerIndex) + ".";
			Layer *regressionLayer = new ConvolutionLayer(cudnnHandle, inputInfo, TensorInfo(4 * numAnchors, inChannels,1,1), PaddingInfo(0,0,1,1),
					dataPath + dataPrefix + "regression.weight.npy", 
					dataPath + dataPrefix + "regression.bias.npy", 
					1, "predict.regression_" + to_string(layerIndex),
					TensorInfo(0,0,0,0, CUDNN_TENSOR_NHWC));
			regressionLayers.push_back(regressionLayer);
			return regressionLayer->getOutputInfo();
		};

		TensorInfo addDwpwLayer(int layerIndex, TensorInfo inputInfo, TensorInfo dwfilterInfo, PaddingInfo dwpaddingInfo,
				TensorInfo pwfilterInfo, PaddingInfo pwpaddingInfo) {
			/* Depthwise */
			string dataPrefix = "backbone." + to_string(layerIndex) + ".depthwise.";
			Layer *dwConvolutionLayer = new ConvolutionTFStyleLayer(cudnnHandle, inputInfo, dwfilterInfo, dwpaddingInfo,
					dataPath + dataPrefix + "weight.npy", "None", dwfilterInfo.c,
					"dwconv_" + to_string(layerIndex));
			backboneLayers.push_back(dwConvolutionLayer);
			/* Use Convolution instead of BatchNormalization */
			Layer *dwBatchNormalizaitionLayer = new DepthwiseConvLayer(cudnnHandle, dwConvolutionLayer->getOutputInfo(),
					TensorInfo(dwfilterInfo.n, dwfilterInfo.c, 1, 1), PaddingInfo(0,0,1,1),	
					dataPath + dataPrefix + "BatchNorm.scale.npy",
					dataPath + dataPrefix + "BatchNorm.bias.npy",
					dwfilterInfo.c, "dwbn_" + to_string(layerIndex));
			backboneLayers.push_back(dwBatchNormalizaitionLayer);

			Layer *dwReluLayer = new ReluLayer(cudnnHandle, dwBatchNormalizaitionLayer->getOutputInfo(), 6.0f,
					"dwrelu_" + to_string(layerIndex));
			backboneLayers.push_back(dwReluLayer);

			/* Pointwise */
			dataPrefix = "backbone." + to_string(layerIndex) + ".pointwise.";
			Layer *pwConvolutionLayer = new ConvolutionLayer(cudnnHandle, dwReluLayer->getOutputInfo(), pwfilterInfo, pwpaddingInfo,
					dataPath + dataPrefix + "weight.npy", 
					dataPath + dataPrefix + "BatchNorm.bias.npy", 
					1, "pwconv_" + to_string(layerIndex));
			backboneLayers.push_back(pwConvolutionLayer);

			Layer *pwReluLayer = new ReluLayer(cudnnHandle, pwConvolutionLayer->getOutputInfo(), 6.0f,
					"pwrelu_" + to_string(layerIndex));
			backboneLayers.push_back(pwReluLayer);

			TensorInfo outputInfo = pwReluLayer->getOutputInfo();
			return pwReluLayer->getOutputInfo();
		};

		void inference(float *input, int imageId, int height, int width) {
			system_clock::time_point start;
			system_clock::time_point end;
			microseconds microSec;
			float *output;
			/* Backbone */
			for(Layer *layer : backboneLayers) {
				start = system_clock::now();
				output = layer->forward(input);
				end = system_clock::now();
				microSec = duration_cast<microseconds>(end - start);
				input = output;
				if(layer->getLayerName().compare("pwrelu_11") == 0 ||
				   layer->getLayerName().compare("pwrelu_13") == 0) {
					blockInputs.push_back(output);
				}
			}

			/* Block */
			input = blockInputs[1];
			for(Layer *layer : blockLayers) {
				start = system_clock::now();
				output = layer->forward(input);
				end = system_clock::now();
				microSec = duration_cast<microseconds>(end - start);
				input = output;
				if(layer->getLayerName().find("relu2") != string::npos)
					blockInputs.push_back(output);
			}
			
			/* Prediction */
			int index = 0;
			int infoIdx = 0;
			input = blockInputs[index++];
			for(Layer *layer : classificationLayers) {
				start = system_clock::now();
				output = layer->forward(input);
				end = system_clock::now();
				microSec = duration_cast<microseconds>(end - start);
				input = output;
				if(layer->getLayerName().find("sigmoid") != string::npos) {
					classificationOutputs.push_back(output);
					input = blockInputs[index++];
				}
			}
			index = 0;
			infoIdx = 0;
			for(Layer *layer : regressionLayers) {
				start = system_clock::now();
				output = layer->forward(blockInputs[index++]);
				end = system_clock::now();
				microSec = duration_cast<microseconds>(end - start);
				regressionOutputs.push_back(output);
			}

			/* Concat */
			index = 0;
			input = classificationOutputs[index++];
			for(Layer *layer : classifiConcatLayers) {
				start = system_clock::now();
				output = layer->func(input, classificationOutputs[index++]);
				end = system_clock::now();
				microSec = duration_cast<microseconds>(end - start);
				input = output;
			}
			scores = output;

			index = 0;
			input = regressionOutputs[index++];
			for(Layer *layer : regressionConcatLayers) {
				start = system_clock::now();
				output = layer->func(input, regressionOutputs[index++]);
				end = system_clock::now();
				microSec = duration_cast<microseconds>(end - start);
				input = output;
			}
			relCodes= output;

			/* Decode boxes */
				start = system_clock::now();
			cuda_decode_boxes(relCodes, prior, predBoxes, 1, totalRowSize, numBoxes);
				end = system_clock::now();
				microSec = duration_cast<microseconds>(end - start);
		
			/* Filter boxes */
				start = system_clock::now();
			checkCUDA(cudaMemcpy(scoresHost, scores, sizeof(float) * (totalRowSize * numClasses), cudaMemcpyDeviceToHost));
			checkCUDA(cudaMemcpy(boxesHost, predBoxes, sizeof(float) * (totalRowSize * numBoxes), cudaMemcpyDeviceToHost));
				end = system_clock::now();
				microSec = duration_cast<microseconds>(end - start);
			filteredResult.clear();
				start = system_clock::now();
			filter_result_ver1(scoresHost, boxesHost, filteredResult, numClasses, numBoxes, totalRowSize);
				end = system_clock::now();
				microSec = duration_cast<microseconds>(end - start);

				start = system_clock::now();
			for(auto r : filteredResult) {
				float boxes[4];
				float score = scoresHost[ (r.second * numClasses) + r.first];
				/* rescale_boxes method */
				for(int i = 0; i < numBoxes; ++i) {
					int scale = (i % 2 == 0) ? (width) : (height);
					boxes[i] = boxesHost[r.second * numBoxes + i] * scale;
				}
				/* convert predictions from xyxsy to xywh */
				boxes[3] -= boxes[1];
				boxes[2] -= boxes[0];
				results.push_back(make_tuple(imageId, r.first, boxes[0], boxes[1], boxes[2], boxes[3], score));
			}
				end = system_clock::now();
				microSec = duration_cast<microseconds>(end - start);
		}

		/* Util */
		void loadInputInfo(vector<tuple<int, int, int>> &inputMetaInfo) {
			const string inputInfoPath = inputPath +  "input_meta.txt";
			ifstream inFile(inputInfoPath);
			string strVal;
			int imageId;
			int height;
			int width;
			while(!inFile.eof()) {
				inFile >> strVal;	imageId = stoi(strVal);
				inFile >> strVal;	height 	= stoi(strVal);
				inFile >> strVal;	width 	= stoi(strVal);
				inputMetaInfo.push_back(make_tuple(imageId, height, width));
			}
			inFile.close();
			return;
		}

		float* loadImage(int index) {
			const string imagePath = inputPath +  "images/";
			const int imageBytes = channel * width * height * sizeof(float);
			NPYParser parser;
			parser.load(imagePath + "image_" + to_string(index) + ".npy");
			parser.parse();
			cudaMemcpy(inputImage, parser.getDataStartAddr(), imageBytes, cudaMemcpyHostToDevice);
			return inputImage;
		}

		void saveResults(void) {
			ofstream outputFile("./data/results.txt");
			string outputString;
			if(outputFile.is_open()) {
				for(auto r : results) {
					outputString = to_string(get<0>(r)) + " " + to_string(get<1>(r)) + " " +\
						to_string(get<2>(r)) + " " + to_string(get<3>(r)) + " " +\
						to_string(get<4>(r)) + " " + to_string(get<5>(r)) + " " +\
						to_string(get<6>(r));
					outputFile << outputString << endl;
				}
			}
			outputFile.close();
		}

	private:
		cudnnHandle_t cudnnHandle;;
		vector<Layer*> network;
		vector<Layer*> backboneLayers;
		vector<Layer*> classificationLayers;
		vector<Layer*> regressionLayers;
		vector<Layer*> classifiConcatLayers;
		vector<Layer*> regressionConcatLayers;
		vector<Layer*> blockLayers;
		vector<float*> blockInputs;
		vector<float*> classificationOutputs;
		vector<float*> regressionOutputs;
		vector<pair<int, int>> filteredResult;
		vector<tuple<int, int, float, float, float, float, float>> results;
		float* prior;
		float* relCodes;
		float* scores;
		float* predBoxes;
		float* scoresHost;
		float* boxesHost;
		int totalRowSize;
		int numClasses;
		int numBoxes;

		string inputPath;
		string dataPath;
		float* inputImage;
		int channel;
		int height;
		int width;
};

int main(void)
{
	system_clock::time_point start;
	system_clock::time_point end;
	microseconds microSec;
	vector<microseconds> inferenceMicroSec;
	microseconds setupTime = microseconds(0);
	microseconds detectTime = microseconds(0);
	int imagesProceed = 0;
	vector<tuple<int, int, int>> inputInfo;
	int batchCount = 500;
	const string inputInfoPath = "./data/coco-2017-val/input_meta.txt";

	/* Init  */
	SSD_MobileNet_v1_224 ssd_mobileNet;
	start = system_clock::now();
	ssd_mobileNet.init();
	ssd_mobileNet.loadInputInfo(inputInfo);
	end = system_clock::now();
	setupTime = duration_cast<microseconds>(end - start);
	cout << "Initialization is done" << endl;
	
	/* Inference */
	for(int i = 0; i < batchCount; ++i) {
		float *image = ssd_mobileNet.loadImage(i);
		start = system_clock::now();
		ssd_mobileNet.inference(image, get<0>(inputInfo[i]), get<1>(inputInfo[i]), get<2>(inputInfo[i])); 
		end = system_clock::now();
		microSec = duration_cast<microseconds>(end - start);
		cout << setw(3) << i + 1 << "/" << batchCount << " Inference time : " << setw(8) << microSec.count() << " us\r" << flush;
		inferenceMicroSec.push_back(microSec);
		if( i > 0 || batchCount == 1) {
			detectTime += microSec;
			imagesProceed++;
		}
	}

	/* Printout */
	cout << "Summary:" << endl;
	cout << "-------------------------------" << endl;
	cout << "Setup time " << setupTime.count() << " us" << endl;
	cout << "Average detection time: " << detectTime.count() / imagesProceed << " us" << endl;

	/* Save result to send COCO API */
	ssd_mobileNet.saveResults();
	return 0;
}
