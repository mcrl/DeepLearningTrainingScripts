#ifndef _INTERACTION_H_
#define _INTERACTION_H_

#include "configs.h"
#include "tensor.h"

__global__ void interaction_forward_kernel (float *d, float **e, float *out, int cnt_e, int batch_size, int vector_size, int output_size);
__global__ void interaction_backward_kernel (float *d, float *dgrad, float **e, float **egrad,
                                             float *outgrad, int cnt_e, int batch_size, int vector_size, int output_size);

class InteractionLayer {
public:
    int batch_size, vector_size, numSparse;
    int outputSize;
    int ndev;

    float **d_sparse, **d_sparse_grad;
    float **h_sparse, **h_sparse_grad;

    InteractionLayer (int vector_size_, int numSparse_, int outputSize_, int batch_size_, int ndev_);

    void forward (Tensor *t_dense, Tensor *t_sparse[], Tensor *t_out);

    void backward (Tensor *t_dense, Tensor *t_dense_grad, 
                   Tensor *t_sparse[], Tensor *t_sparse_grad[],
                   Tensor *t_out, Tensor *t_out_grad );
};

#endif // _INTERACTION_H_
