#include <stdlib.h>
#include <stdbool.h>

#include <cuda.h>
#include <cudnn.h>
#include <cublas_v2.h>

#include "memory.h"

static size_t size_of_cudnn_data_type(cudnnDataType_t data_type)
{
  switch (data_type) {
    case CUDNN_DATA_FLOAT: return (size_t)4;
    default: return (size_t)0;
  }
}

static inline int distribute_batch(int n, int dev)
{
  return (n / num_devices) + (dev < num_devices);
}

void create_mem_data(
    gpu_mem mem, cudnnDataType_t data_type,
    int n, int c, int h, int w)
{
  mem->data_type = data_type;
  mem->mem_type = DATA;

  mem->reserved = true;
  mem->distributed = true;
  mem->consistent = false;

  for (int dev = 0; dev < num_devices; dev++) {
    chkCUDA(cudaSetDevice(dev));

    chkCUDNN(cudnnCreateTensorDescriptor(&mem->tensorDesc[dev]));
    mem->filter_desc = NULL;

    int n_dev = distribute_batch(n, dev);

    chkCUDNN(cudnnSetTensor4dDescriptor(
          mem->tensorDesc[dev], CUDNN_TENSOR_NCHW,
          data_type, n_dev, c, h, w));

    size_t size_per_dev =
      size_of_cudnn_data_type(data_type) * n_dev * c * h * w;

    chkCUDA(cudaMalloc(&mem->dev_ptr[dev], size_per_dev));
  }
}

void create_mem_data_gradient(
    gpu_mem mem, cudnnDataType_t data_type,
    int n, int c, int h, int w)
{
  mem->data_type = data_type;
  mem->mem_type = DATA_GRADIENT;

  mem->reserved = false;
  mem->distributed = true;
  mem->consistent = false;

  for (int dev = 0; dev < num_devices; dev++) {
    chkCUDA(cudaSetDevice(dev));

    chkCUDNN(cudnnCreateTensorDescriptor(&mem->tensorDesc[dev]));
    mem->filter_desc = NULL;

    int n_dev = distribute_batch(n, dev);

    chkCUDNN(cudnnSetTensor4dDescriptor(
          mem->tensorDesc[dev], CUDNN_TENSOR_NCHW,
          data_type, n_dev, c, h, w));

    size_t size_per_dev =
      size_of_cudnn_data_type(data_type) * n_dev * c * h * w;

    chkCUDA(cudaMalloc(&mem->dev_ptr[dev], size_per_dev));
  }
}

void create_mem_param(
    gpu_mem mem, cudnnDataType_t data_type,
    int k, int c, int h, int w)
{
  mem->data_type = data_type;
  mem->mem_type = PARAM;

  mem->reserved = true;
  mem->distributed = true;
  mem->consistent = false;

  // TODO
  for (int dev = 0; dev < num_devices; dev++) {
    chkCUDA(cudaSetDevice(dev));

    chkCUDNN(cudnnCreateTensorDescriptor(&mem->tensorDesc[dev]));
    mem->filter_desc = NULL;

    int n_dev = distribute_batch(n, dev);

    chkCUDNN(cudnnSetTensor4dDescriptor(
          mem->tensorDesc[dev], CUDNN_TENSOR_NCHW,
          data_type, n_dev, c, h, w));

    size_t size_per_dev =
      size_of_cudnn_data_type(data_type) * n_dev * c * h * w;

    chkCUDA(cudaMalloc(&mem->dev_ptr[dev], size_per_dev));
  }
}
