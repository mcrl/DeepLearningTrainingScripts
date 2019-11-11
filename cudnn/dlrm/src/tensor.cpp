#include "tensor.h"

void Tensor::initialize () {
    CUDA_CALL( cudaMalloc(&d_mem, bytes) );
    CUDNN_CALL( cudnnCreateTensorDescriptor(&desc) );
}

Tensor::Tensor () : bytes(0), N(0), C(0), H(0), W(0), ndev(0) 
{
}

Tensor::Tensor (int N_, int C_, int H_, int W_, int ndev_) :
        N(N_), C(C_), H(H_), W(W_), ndev(ndev_)
{
    CUDA_CALL( cudaSetDevice(ndev) );

    bytes = N * C * H * W * sizeof(float);
    initialize();
    CUDNN_CALL( cudnnSetTensor4dDescriptor(desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
        N, C, H, W) );
    if( DEBUG ) cudaDeviceSynchronize();
}

Tensor::~Tensor () {
}

void Tensor::hostToDevice (float *h_mem, int bytes) {

    CUDA_CALL( cudaSetDevice(ndev) );

    CUDA_CALL( cudaMemcpy(d_mem, h_mem, bytes, cudaMemcpyHostToDevice) );
    if( DEBUG ) cudaDeviceSynchronize();
}

void Tensor::deviceToHost (float *h_mem, int bytes) {

    CUDA_CALL( cudaSetDevice(ndev) );

    CUDA_CALL( cudaMemcpy(h_mem, d_mem, bytes, cudaMemcpyDeviceToHost) );
    if( DEBUG ) cudaDeviceSynchronize();
}

void IntegerTensor::initialize () {
    CUDA_CALL( cudaMalloc(&d_mem, bytes) );
    CUDNN_CALL( cudnnCreateTensorDescriptor(&desc) );
}

IntegerTensor::IntegerTensor () : bytes(0), N(0), C(0), H(0), W(0) 
{
}

IntegerTensor::IntegerTensor (int N_, int C_, int H_, int W_, int ndev_) :
        N(N_), C(C_), H(H_), W(W_), ndev(ndev_)
{

    CUDA_CALL( cudaSetDevice(ndev) );

    bytes = N * C * H * W * sizeof(int);
    initialize();
    CUDNN_CALL( cudnnSetTensor4dDescriptor(desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_INT32,
        N, C, H, W) );
    if( DEBUG ) cudaDeviceSynchronize();
}

IntegerTensor::~IntegerTensor () {
}

void IntegerTensor::hostToDevice (int *h_mem, int bytes) {

    CUDA_CALL( cudaSetDevice(ndev) );

    CUDA_CALL( cudaMemcpy(d_mem, h_mem, bytes, cudaMemcpyHostToDevice) );
    if( DEBUG ) cudaDeviceSynchronize();
    
}

void IntegerTensor::deviceToHost (int *h_mem, int bytes) {

    CUDA_CALL( cudaSetDevice(ndev) );

    CUDA_CALL( cudaMemcpy(h_mem, d_mem, bytes, cudaMemcpyDeviceToHost) );
    if( DEBUG ) cudaDeviceSynchronize();
}


