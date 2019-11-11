#ifndef _TENSOR_H_
#define _TENSOR_H_

#include "configs.h"

class Tensor {
public:
    cudnnTensorDescriptor_t desc;
    float *d_mem;
    int bytes;
    int N, C, H, W;
    int ndev;


    void initialize ();
    Tensor ();
    Tensor (int N_, int C_, int H_, int W_, int ndev_);
    ~Tensor ();

    void hostToDevice (float *h_mem, int bytes);
    void deviceToHost (float *h_mem, int bytes);
};

class IntegerTensor {
public:
    cudnnTensorDescriptor_t desc;
    int *d_mem;
    int bytes;
    int N, C, H, W;
    int ndev;

    void initialize ();
    IntegerTensor ();
    IntegerTensor (int N_, int C_, int H_, int W_, int ndev_);
    ~IntegerTensor ();

    void hostToDevice (int *h_mem, int bytes);
    void deviceToHost (int *h_mem, int bytes);
};

#endif // _TENSOR_H
