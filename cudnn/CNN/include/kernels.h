#ifndef _KERNELS_H_
#define _KERNELS_H_

__global__ void set_label(
    int batch_size, int class_cnt, int *label_in, float *label);

__global__ void concat2(
    int batch_size, int channel1, int channel2, int height, int width,
    int fwd, float *in1, float *in2, float *out);

#endif // _KERNELS_H_
