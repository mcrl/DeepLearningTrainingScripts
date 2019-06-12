#ifndef _MEMORY_H_
#define _MEMORY_H_

#include <stdlib.h>
#include <stdbool.h>

#include <cuda.h>
#include <cudnn.h>
#include <cublas_v2.h>

#define MAX_DEV 1
int num_devices = 1;

struct _gpu_mem {
  void *dev_ptr[MAX_DEV];

  int batch;
  int channel;
  int height;
  int width;

  cudnnDataType_t data_type;

  cudnnTensorDescriptor_t tensor_desc[MAX_DEV];
  cudnnFilterDescriptor_t filter_desc;

  enum {
    EMPTY,
    DATA,
    DATA_GRADIENT,
    PARAM,
    PARAM_GRADIENT,
    WORKSPACE,
    RESERVESPACE,
    INVALID
  } mem_type;

  bool reserved;
  bool distributed;
  bool consistent;
};

typedef _gpu_mem *gpu_mem;

void create_mem_data(
    gpu_mem mem, cudnnDataType_t data_type,
    int n, int c, int h, int w);

void create_mem_data_gradient(
    gpu_mem mem, cudnnDataType_t data_type,
    int n, int c, int h, int w);

void create_mem_param(
    gpu_mem mem, cudnnDataType_t data_type,
    int k, int c, int h, int w);

#endif // _MEMORY_H_
