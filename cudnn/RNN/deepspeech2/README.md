# deepspeech.cudnn
Implementation of DeepSpeech2 with CUDA and CUDA libraries.

## Installation
The Warp-CTC library by baidu is located in warp-ctc folder and needed to be installed for training.  
Checkout the warp-ctc compilation guide at [**warp-ctc**](https://github.com/baidu-research/warp-ctc).  
After install Warp-CTC, modify the Makefile to correctly link CUDA and cuDNN libraries.  
```
CC_CUDA_INCLUDE := -I/usr/local/cuda/include -I/usr/local/cuda/cudnn-7.4.2/include
CC_CUDA_LIBRARY := -L/usr/local/cuda/cudnn-7.4.2/lib64 -L/usr/local/cuda/lib64/ -lcudnn -lcudart -lcublas -lnccl
```
Those two lines may need to be modified.

To compile the application, a single make command will do all the things.
```
make
```

## Datasets

To download and preprocess datasets (e.g. LibriSpeech), use [**the PyTorch reference**] `pytorch/RNN/deepspeech2`.
Uncomment the lines 221 ~ 246 and do the following:
```
cd deepspeech.pytorch/data
python librispeech.py
cd ..
python train.py --train-manifest data/librispeech_train_manifest.csv --val-manifest data/librispeech_val_manifest.csv
```

## Hyperparameters
The values of hyperparameters are defined in include/params.h.  
Recommend that only
+ the number of epochs (params.epochs)
+ batch size for training (params.batch_size)
+ batch size for inference (params.batch_size_eval)  

those be modified.

## How to run
The application takes at most 4 arguments.  
The number of GPUs per node, train or infer to specify either training or inference, and a pair of dataset to use.  
The application can be runned with MPI library. Checkout run.sh and hosts file to see details.  
  
To run with single GPU and train with default AN4 dataset:
```
./deepspeech.cudnn 1 train
```
or just
```
./deepspeech.cudnn 1
```

To run with 2 GPUs in the node and inference with a custom dataset:
```
./deepspeeh.cudnn 2 infer data/custom_wav.bin data/custom_txt.bin
```

To run with 2 GPUs per node and train with default AN4 dataset:
```
mpirun -x LD_LIBRARY_PATH -npernode 2 --hostfile hosts -H c0,c1 ./deepspeech.cudnn 2
```
or just
```
./run.sh 2 c0,c1
```
