#ifndef _KERNELS_H_
#define _KERNELS_H_

#ifdef __cplusplus
#define EXTERNC extern "C"
#else
#define EXTERNC
#endif
EXTERNC void cuda_shuffle(float *in, float *out, int N, int G, int C, int HW);
EXTERNC void cuda_concatenate(float *in, float *in2, float *out, int N, int C, int C2, int HW);
EXTERNC void cuda_pad(float *in, float *out, int N, int C, int srcH, int srcW, int targetH, int targetW);
EXTERNC void cuda_decode_boxes(float *rel_codes, float *boxes, float *pred_boxes, int N, int C, int HW);
EXTERNC void cuda_prelu(float *in, float* out, float *sigma, int N, int C, int HW);
EXTERNC void cuda_depthwise_conv(float *in, float *out, float *filter, float *bias, int C, int srcH, int srcW, int dstH, int dstW, int stride, int offset, int filter_size);
#endif

