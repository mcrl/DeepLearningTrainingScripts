#ifndef _EXECUTE_H_
#define _EXECUTE_H_

#include <stdlib.h>
#include <stdbool.h>

#include <cuda.h>
#include <cudnn.h>
#include <cublas_v2.h>

#include "memory.h"

////////////////////////////////////////////////////////////
// Executer Management API
////////////////////////////////////////////////////////////

int __init_stream_executer(void);

int __finalize_stream_executer(void);

////////////////////////////////////////////////////////////
// cuDNN based API
////////////////////////////////////////////////////////////

/* Activation */
int execute_actv_bwd(
    cudnnActivationDescriptor_t actvDesc,
    gpu_mem y, gpu_mem dy, gpu_mem x, gpu_mem dx);

int execute_actv_fwd(
    cudnnActivationDescriptor_t actvDesc,
    gpu_mem x, gpu_mem y);

/* Bias */
int execute_bias_bwd(gpu_mem dy, gpu_mem db);

int execute_bias_fwd(gpu_mem b, gpu_mem y);

/* Convolution */
int execute_conv_bwd_data(
    cudnnConvolutionDescriptor_t convDesc,
    cudnnConvolutionBwdDataAlgo_t algo,
    gpu_mem w, gpu_mem dy,
    gpu_mem dx, gpu_mem workSpace);

int execute_conv_bwd_filter(
    cudnnConvolutionDescriptor_t convDesc,
    cudnnConvolutionBwdFilterAlgo_t algo,
    gpu_mem x, gpu_mem dy,
    gpu_mem dw, gpu_mem workSpace);

int execute_conv_fwd(
    cudnnConvolutionDescriptor_t convDesc,
    cudnnConvolutionFwdAlgo_t algo,
    gpu_mem x, gpu_mem w,
    gpu_mem y, gpu_mem workSpace);

/* Element-wise Operation */
int execute_elt_bwd(
    cudnnOpTensorDescriptor_t opDesc,
    gpu_mem y1, gpu_mem y2, gpu_mem x);

int execute_elt_fwd(
    cudnnOpTensorDescriptor_t opDesc,
    gpu_mem x1, gpu_mem x2, gpu_mem y);

/* Pooling */
int execute_pool_bwd(
    cudnnPoolingDescriptor_t poolDesc,
    gpu_mem y, gpu_mem dy, gpu_mem x, gpu_mem dx);

int execute_pool_fwd(
    cudnnPoolingDescriptor_t poolDesc,
    gpu_mem x, gpu_mem y);

/* Softmax */
int execute_softmax_bwd(
    cudnnSoftmaxAlgorithm_t algo,
    cudnnSoftmaxMode_t mode,
    gpu_mem y, gpu_mem dy, gpu_mem dx);

int execute_softmax_fwd(
    cudnnSoftmaxAlgorithm_t algo,
    cudnnSoftmaxMode_t mode,
    gpu_mem x, gpu_mem y);

////////////////////////////////////////////////////////////
// cuBLAS based API
////////////////////////////////////////////////////////////

/* Update Weight */
int execute_apply_gradient(
    const float learning_rate,
    gpu_mem dw, gpu_mem w);

////////////////////////////////////////////////////////////
// cuda kernel based API
////////////////////////////////////////////////////////////


#endif // _EXECUTE_H_
