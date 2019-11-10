#ifndef __SNUDNN_H_
#define __SNUDNN_H_
#include <nccl.h>

#include <cublas_v2.h>
#include <cudnn.h>
#include <stddef.h>
#include "tensor.h"

extern float _one;
extern float _zero;

extern cudaStream_t data_stream;
extern cudaStream_t update_stream;
extern int rank, nrank;
extern ncclUniqueId id;
extern ncclComm_t comm;

void snudnn_init(int argc, char *argv[]);
void snudnn_memseti(int *m, size_t nelem, int val);
void snudnn_memsetf(float *m, size_t nelem, float val);
void snudnn_uniform(float *m, size_t nelem, float min, float max);


#define FATAL(msg) { printf("%s\n", msg); exit(1); }


extern cublasHandle_t cublas_handle;
extern cudnnHandle_t cudnn_handle;

tensor_t* snudnn_tanh_forward(tensor_t *x);
tensor_t* snudnn_tanh_backward(tensor_t *dy, tensor_t *x, tensor_t *y);

tensor_t* snudnn_softmax0_forward(tensor_t *x);
tensor_t* snudnn_softmax0_backward(tensor_t *dy, tensor_t *y);

#endif //__SNUDNN_H_
