#ifndef _CNN_CUDA_H_
#define _CNN_CUDA_H_

#ifdef __cplusplus
#define EXTERNC extern "C"
#else
#define EXTERNC
#endif

EXTERNC void cublas_linear_fwd(float *input, float *output, float *weight, int batch, int in, int out);
EXTERNC void cublas_linear_bwd_data(float *d_input, float *d_output, float *weight, int batch, int in, int out);
EXTERNC void cublas_linear_bwd_weight(float *input, float *d_output, float *d_weight, int batch, int in, int out);
EXTERNC void cublas_apply_grad(float *x, float *dx, float lr, int cnt);
EXTERNC void cuda_set_label(int batch_size, int class_cnt, int *label_in, float *label);
EXTERNC void cuda_concat(int batch_size, int in_cnt, int *channel_in, int height, int width, int fwd, float **in, float *out);

#endif // _CNN_CUDA_H_
