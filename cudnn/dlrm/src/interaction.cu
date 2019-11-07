#include "interaction.h"

__global__ void interaction_forward_kernel (float *d, float **e, float *out, int cnt_e, int batch_size, int vector_size, int output_size) {
    int batch = blockIdx.x * blockDim.x + threadIdx.x;
    int idx = blockIdx.y * blockDim.y + threadIdx.y;

    if ( batch >= batch_size || idx >= output_size ) return;

    if ( idx < vector_size ) { // d
        out[batch * output_size + idx] = d[batch * vector_size + idx];
    }
    else if( idx < vector_size + cnt_e ) { // de0 de1 ... de25
        float sum = 0.0;
        for (int k = 0; k < vector_size; k++) {
            sum += d[batch * vector_size + k] * e[idx-vector_size][batch * vector_size + k]; 
        }
        out[batch * output_size + idx] = sum;
    }
    else {
        int i, j, t = idx - (vector_size + cnt_e);
        for (i = 0; i < cnt_e; i++) {
            if ( t < cnt_e - i - 1 )  {
                j = t + i + 1;
                break;
            }
            t -= cnt_e - i - 1;
        }

        float sum = 0.0;
        for (int k = 0; k < vector_size; k++) {
            sum += e[i][batch * vector_size + k] * e[j][batch * vector_size + k];
        }
        out[batch * output_size + idx] = sum;
    }

}

__global__ void interaction_backward_kernel (float *d, float *dgrad, float **e, float **egrad,
                                                float *outgrad, int cnt_e, int batch_size, int vector_size, int output_size) {
    int batch = blockIdx.x * blockDim.x + threadIdx.x;
    int idx = blockIdx.y * blockDim.y + threadIdx.y;

    if( batch >= batch_size || idx >= output_size ) return;

    if( idx < vector_size ) { // d
        atomicAdd(dgrad + batch * vector_size + idx, outgrad[batch * output_size + idx]);
    }
    else if( idx < vector_size + cnt_e ) {
        float delta = outgrad[batch * output_size + idx];
        for (int k = 0; k < vector_size; k++) {
            atomicAdd(dgrad + batch * vector_size + k, delta * e[idx - vector_size][batch * vector_size + k]);
            atomicAdd(egrad[idx - vector_size] + batch * vector_size + k, delta * d[batch * vector_size + k]);
        }
    }
    else {
        int i, j, t = idx - (vector_size + cnt_e);
        for (i = 0; i < cnt_e; i++) {
            if ( t < cnt_e - i - 1 )  {
                j = t + i + 1;
                break;
            }
            t -= cnt_e - i - 1;
        }

        float delta =  outgrad[batch * output_size + idx];
        for (int k = 0; k < vector_size; k++) {
            atomicAdd(egrad[i] + batch * vector_size + k, delta * e[j][batch * vector_size + k]);
            atomicAdd(egrad[j] + batch * vector_size + k, delta * e[i][batch * vector_size + k]);
        }
    }
}


InteractionLayer::InteractionLayer (int vector_size_, int numSparse_, int outputSize_, int batch_size_, int ndev_) :
                vector_size(vector_size_), numSparse(numSparse_), outputSize(outputSize_), batch_size(batch_size_), ndev(ndev_)
{
    h_sparse = (float**) malloc( numSparse * sizeof(float*) );
    h_sparse_grad = (float**) malloc( numSparse * sizeof(float*) );
    CUDA_CALL( cudaMalloc(&d_sparse, numSparse * sizeof(float*) ) );
    CUDA_CALL( cudaMalloc(&d_sparse_grad, numSparse * sizeof(float*) ) );
}

void InteractionLayer::forward (Tensor *t_dense, Tensor *t_sparse[], Tensor *t_out) {

    CUDA_CALL( cudaSetDevice(ndev) );

    for (int i = 0; i < numSparse; i++) h_sparse[i] = t_sparse[i]->d_mem;

    CUDA_CALL( cudaMemcpy(d_sparse, h_sparse, numSparse * sizeof(float*), cudaMemcpyHostToDevice) );

    dim3 blocks2((batch_size+15) / 16, (outputSize + 15) / 16 );
    dim3 threadPerBlock2(16, 16);
    interaction_forward_kernel<<<blocks2, threadPerBlock2>>>(t_dense->d_mem, d_sparse, t_out->d_mem, numSparse, batch_size, vector_size, outputSize);
}

void InteractionLayer::backward (Tensor *t_dense, Tensor *t_dense_grad, 
                Tensor *t_sparse[], Tensor *t_sparse_grad[],
                Tensor *t_out, Tensor *t_out_grad ) {

    CUDA_CALL( cudaSetDevice(ndev) );

    // 1. concat sparse
    for (int i = 0; i < numSparse; i++) {
        h_sparse[i] = t_sparse[i]->d_mem;
        h_sparse_grad[i] = t_sparse_grad[i]->d_mem;
    }
    CUDA_CALL( cudaMemcpy(d_sparse, h_sparse, numSparse * sizeof(float*), cudaMemcpyHostToDevice) );
    CUDA_CALL( cudaMemcpy(d_sparse_grad, h_sparse_grad, numSparse * sizeof(float*), cudaMemcpyHostToDevice) );

    // 2. init grads to zero
    CUDA_CALL( cudaMemset(t_dense_grad->d_mem, 0, batch_size * vector_size * sizeof(float)) );
    for (int i = 0; i < numSparse; i++) {
        CUDA_CALL( cudaMemset(t_sparse_grad[i]->d_mem, 0, batch_size * vector_size * sizeof(float)) );
    }

    // 3. backward kernel
    dim3 blocks((batch_size+15) / 16, (outputSize + 15) / 16 );
    dim3 threadPerBlock(16, 16);
    interaction_backward_kernel<<<blocks, threadPerBlock>>>(
                                        t_dense->d_mem, t_dense_grad->d_mem, d_sparse, d_sparse_grad, 
                                        t_out_grad->d_mem, numSparse, batch_size, vector_size, outputSize);
}
