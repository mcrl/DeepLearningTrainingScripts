#ifndef _EXECUTE_H_
#define _EXECUTE_H_

#include <stdlib.h>
#include <stdbool.h>

#include <cuda.h>
#include <cudnn.h>
#include <cublas_v2.h>

#ifdef __cplusplus
extern "C" {
#endif // __cplusplus

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
int execute_act_bwd(
    cudnnActivationDescriptor_t actDesc,
    gpu_mem y, gpu_mem dy, gpu_mem x, gpu_mem dx);

int execute_act_fwd(
    cudnnActivationDescriptor_t actDesc,
    gpu_mem x, gpu_mem y);

/* Batch Normalization */
int execute_bn_bwd(
    cudnnBatchNormMode_t mode,
    double eps,
    gpu_mem x, gpu_mem dy, gpu_mem dx,
    gpu_mem w, gpu_mem dw, gpu_mem db,
    gpu_mem s_mean, gpu_mem s_var);

int execute_bn_fwd(
    cudnnBatchNormMode_t mode,
    double eaf, double eps,
    gpu_mem x, gpu_mem y, gpu_mem w, gpu_mem b,
    gpu_mem r_mean, gpu_mem r_var,
    gpu_mem s_mean, gpu_mem s_var);

/* Bias */
int execute_bias_bwd(gpu_mem dy, gpu_mem db);

int execute_bias_fwd(gpu_mem b, gpu_mem y);

/* Branch */
int execute_branch_bwd(
    cudnnOpTensorDescriptor_t opDesc,
    int fan_out, gpu_mem dy[], gpu_mem dx);

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

int execute_get_conv_bwd_data_algo(
    cudnnConvolutionDescriptor_t convDesc,
    gpu_mem w, gpu_mem dy, gpu_mem dx,
    cudnnConvolutionBwdDataAlgo_t *algo);

int execute_get_conv_bwd_data_ws_size(
    cudnnConvolutionDescriptor_t convDesc,
    cudnnConvolutionBwdDataAlgo_t algo,
    gpu_mem w, gpu_mem dy, gpu_mem dx,
    size_t *ws_size);

int execute_get_conv_bwd_filter_algo(
    cudnnConvolutionDescriptor_t convDesc,
    gpu_mem x, gpu_mem dy, gpu_mem dw,
    cudnnConvolutionBwdFilterAlgo_t *algo);

int execute_get_conv_bwd_filter_ws_size(
    cudnnConvolutionDescriptor_t convDesc,
    cudnnConvolutionBwdFilterAlgo_t algo,
    gpu_mem x, gpu_mem dy, gpu_mem dw,
    size_t *ws_size);

int execute_get_conv_fwd_algo(
    cudnnConvolutionDescriptor_t convDesc,
    gpu_mem x, gpu_mem w, gpu_mem y,
    cudnnConvolutionFwdAlgo_t *algo);

int execute_get_conv_fwd_ws_size(
    cudnnConvolutionDescriptor_t convDesc,
    cudnnConvolutionFwdAlgo_t algo,
    gpu_mem x, gpu_mem w, gpu_mem y,
    size_t *ws_size);

/* Element-wise Operation */
int execute_elt(
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

/* Linear */
int execute_linear_bwd_data(gpu_mem w, gpu_mem dy, gpu_mem dx);

int execute_linear_bwd_weight(gpu_mem x, gpu_mem dy, gpu_mem dw);

int execute_linear_fwd(gpu_mem x, gpu_mem w, gpu_mem y);

/* Update Weight */
int execute_apply_gradient(
    const float learning_rate, gpu_mem dw, gpu_mem w);

////////////////////////////////////////////////////////////
// CUDA kernel based API
////////////////////////////////////////////////////////////

/* Concatenation */
int execute_concat_bwd(int fan_in, gpu_mem dy, gpu_mem dx[]);

int execute_concat_fwd(int fan_in, gpu_mem x[], gpu_mem y);

/* Softmax */
int execute_set_label(gpu_mem l, gpu_mem dy);

////////////////////////////////////////////////////////////
// CUDA runtime based API
////////////////////////////////////////////////////////////

int synch_comp(void);

int synch_comm(void);

int synch_device(void);

#ifdef __cplusplus
}
#endif // __cplusplus

#endif // _EXECUTE_H_
