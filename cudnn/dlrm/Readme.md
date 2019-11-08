# How to Run

Prepare the training data [data file](https://labs.criteo.com/2014/09/kaggle-contest-dataset-now-available-academic-use/) as ```./data/data.txt```
(Note that the test data is not used, because we cannot check the answer. It is for the [kaggle competition](https://www.kaggle.com/c/criteo-display-ad-challenge/data))

Execute the program with following command.

```
make
mpirun --host c0,c1,c2,c3 -x CUDA_VISIBLE_DEVICES=0,1,2,3 ./dlrm <# nodes> <# GPUs per node> <batch size> <epochs> <learning rate>
```

Data preprocessing will take about ~1 hour.

After first run, preprocessed data will be saved as ```./data/processed.txt``` and ```feature_map.txt```.

# About model

## Dataset ([reference](https://www.kaggle.com/c/criteo-display-ad-challenge/data))

[Criteo Kaggle Display Advertising Challenge Dataset](https://labs.criteo.com/2014/09/kaggle-contest-dataset-now-available-academic-use/)
contains a protion of Criteo's traffic over a period of 7 days. Each row corresponds to a display ad served by Criteo. Positive (clicked) and negatives (non-clicked) examples have both been subsampled at different rates in order to reduce the dataset size. The examples are chronologically ordered. 

The first column denotes whether if an ad was clicked(1) or not(0).

Next 13 columns are dense(integer) features. These are mostly count features.

Next 26 columns are sparse features. The values of these features have been hashed onto 32 bits for anonymization purposes. 

## Preprocessing

Preprocessing is implemented at ```include/data.h``` and ```src/data.cpp```

- Dense features are log-transformed. Each dense feature value x is be transformed to log(x+1). Negative valued x's are omitted and thus transformed to log(1) = 0.

- Sparse features are transformed to indicies in order to be given as inputs of embedding layers.

Preprocessed data are stored at ```data/processed.txt```. Mapping information of sparse features are stored at ```data/feature_map.txt```.

## Model structure

```
output:
                    probability of a click
model:                        |
                     top FC  /\
                            /__\
                              |
      _______________ _> Interaction  <______________
    /                         |                      \
   /\   bot FC                |                       |
  /__\                        |           ...         |
   |                          |                       |
   |                         Op                      Op
   |                    ____/__\_____           ____/__\____
   |                   |_Emb_|____|__|    ...  |_Emb_|__|___|
input:
[ dense features ]     [sparse indices] , ..., [sparse indices]
```

In this model, there are 13 dense features, and 26 sparse indices. Hence, first input size of the bottom FC is 13, and number of embedding tables are 26.

# Files

**```src/main.cpp```**

**```include/activation.h```, ```src/activation.cpp```**

- Activation layer using cuDNN.

**```include/data.h```, ```src/data.cpp```**

- Data parsing and preprocessing.

- Data loading functionalities.

**```include/embedding.h```, ```src/embedding.cu```**

- Embedding layer with CUDA kernel.

**```include/embeddingbag.h```, ```src/embeddingbag.cu```**

- EmbeddingBag layer with CUDA kernel.

**```include/fc.h```, ```src/fc.cpp```**

- Fully connected (Linear) layer using cuDNN.

**```include/interaction.h```, ```src/interaction.cu```**

- Interaction (DLRM-specific layer) with CUDA kernel.

**```include/tensor.h```, ```src/tensor.cpp```**

- cuDNN tensor object wrapper.

**```include/timer.h```, ```src/timer.cpp```**

- Simple timer utility.

**```include/utils.h```, ```src/utils.cpp```**

- Some utility functions.

# References

[DLRM: An advanced, open source deep learning recommendation model](https://ai.facebook.com/blog/dlrm-an-advanced-open-source-deep-learning-recommendation-model/)

[Deep Learning Recommendation Model for Personalization and Recommendation Systems](https://arxiv.org/abs/1906.00091)



[DLRM github repository](https://github.com/facebookresearch/dlrm)