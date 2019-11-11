#include "embeddingbag.h"

__global__ 
void embeddingbag_forward_kernel (int *in, float *table, float *out, int batch_size, int vector_size, int bag_size) {
    int batch = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if ( batch >= batch_size || y >= vector_size ) return;

    int low = batch * bag_size, high = (batch + 1) * bag_size;
    float sum = 0.0;
    for (int i = low; i < high; i++) {
        if ( in[i] == -1 ) continue;
        sum += table[in[i] * vector_size + y];
    }
    out[batch * vector_size + y] = sum;
}

__global__ 
void embeddingbag_backward_kernel (int *in, float *table, float *out_grad, int batch_size, int vector_size, int bag_size, float lr) {
    int batch = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if ( batch >= batch_size || y >= vector_size ) return;

    int low = batch * bag_size, high = (batch + 1) * bag_size;
    for (int i = low; i < high; i++) {
        if ( in[i] == -1 ) continue;
        atomicAdd(table + in[i] * vector_size + y, lr * out_grad[batch * vector_size + y]);
    }
}

EmbeddingBag::EmbeddingBag (int batch_size_, int rows_, int bag_size_, int vector_size_, int ndev_, bool init) :
                batch_size(batch_size_), rows(rows_), bag_size(bag_size_), vector_size(vector_size_), ndev(ndev_)
{
    CUDA_CALL( cudaMalloc(&table, rows * vector_size * sizeof(float)) );
    CUDA_CALL( cudaMalloc(&gatheredIn, batch_size * bag_size * NDEV * NNODE * sizeof(int)) );
    CUDA_CALL( cudaMalloc(&gatheredDelta, batch_size * NDEV * NNODE * vector_size * sizeof(float)) );
    if ( init ) initRandUniform(table, rows * vector_size, sqrt(1.0 / rows), ndev);
}

void EmbeddingBag::forward (IntegerTensor *t_in, Tensor *t_out) {

    CUDA_CALL( cudaSetDevice(ndev) );

    dim3 blocks((batch_size+15) / 16, (vector_size+15) / 16);
    dim3 threadPerBlock(16, 16);
    embeddingbag_forward_kernel<<<blocks, threadPerBlock>>>(t_in->d_mem, table, t_out->d_mem, batch_size, vector_size, bag_size);
}

void EmbeddingBag::backward (IntegerTensor *t_in, Tensor *t_out, Tensor *t_out_grad) {
    in = t_in->d_mem;
    delta = t_out_grad->d_mem;
}

void EmbeddingBag::update () {
    CUDA_CALL( cudaSetDevice(ndev) );

    dim3 blocks((batch_size * NDEV * NNODE + 15) / 16, (vector_size + 15) / 16);
    dim3 threadPerBlock(16, 16);
    embeddingbag_backward_kernel<<<blocks, threadPerBlock>>>(gatheredIn, table, gatheredDelta, batch_size, vector_size, bag_size, lr);
}

