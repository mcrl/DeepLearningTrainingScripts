# deepspeech.cudnn
Implementation of DeepSpeech2 with CUDA and CUDA libraries.

## Installation
The Warp-CTC library by baidu is located in warp-ctc folder and needed to be installed for training.  
Checkout the warp-ctc compilation guide at [**warp-ctc**](https://github.com/baidu-research/warp-ctc).  
Makefile may have to be modified to link CUDA and cuDNN libraries.  

## Datasets
There is a preprocessed AN4 dataset in data folder.  
To download and preprocess with another datasets (e.g. LibriSpeech),  
use [**the PyTorch reference**](https://github.com/SeanNaren/deepspeech.pytorch) and replace the train.py with scripts/train.py.  
Uncomment the line 221 ~ 246 and run the PyTorch reference with replaced train.py.  

## Hyperparameters
The values of hyperparameters are defined in include/params.h.  
Recommend that only
+ the number of epochs(params.epochs)
+ batch size for training(params.batch_size)
+ batch size for inference(params.batch_size_eval)
those be modified.

## How to run
The application takes at most 4 arguments.  
The number of GPUs per node, train or infer to specify either training or inference, and a pair of dataset to use.  
The application can be runned with MPI library. Checkout run.sh and hosts file to see details.  

