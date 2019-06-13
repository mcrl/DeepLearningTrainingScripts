#include <stdlib.h>
#include <stdbool.h>

#include <cuda.h>
#include <cudnn.h>
#include <nccl.h>

#include "memory.h"
#include "utils.h"

//extern int node_id;
//extern int num_nodes;
//extern int num_workers;
int num_devices = 1;
ncclUniqueId nccl_id;

static cudaStream_t memory_stream[MAX_NDEV];
static ncclComm_t nccl_comm[MAX_NDEV];

// FIXME: duplicated code
static size_t size_of_cudnn_data_type[] = { 4, 8, 2, 1, 4, 4, 1, 4, 32 };

static inline int distribute(int n, int dev)
{
  return (n / num_devices) + (dev < n % num_devices);
}

////////////////////////////////////////////////////////////
// Memory Object Management API
////////////////////////////////////////////////////////////

int __init_object_manager()
{
  static bool initialized = false;

  if (initialized) return -1;

  ncclGroupStart();

  for (int dev = 0; dev < num_devices; dev++) {
    chkCUDA(cudaSetDevice(dev));
    chkCUDA(cudaStreamCreate(&memory_stream[dev]));
    ncclCommInitRank(&nccl_comm[dev], 1 * num_devices, nccl_id, 0 * num_devices + dev);
  }

  ncclGroupEnd();

  initialized = true;

  return 0;
}

int __finalize_object_manager()
{
  for (int dev = 0; dev < num_devices; dev++) {
    chkCUDA(cudaSetDevice(dev));
    chkCUDA(cudaStreamDestroy(memory_stream[dev]));
    ncclCommDestroy(nccl_comm[dev]);
  }

  return 0;
}

static int create_4d_tensor(
    gpu_mem mem, cudnnDataType_t data_type, int n, int c, int h, int w);

/* int create_buffer[DATA](gpu_mem *, int, cudnnDataType, [int]) */
int create_buffer_data(gpu_mem *mem, int ndim, ...)
{
  int args[CUDNN_DIM_MAX];
  va_list ap;

  va_start(ap, ndim);
  cudnnDataType_t data_type = (cudnnDataType_t)va_arg(ap, int);
  for (int i = 0; i < ndim; i++) {
    args[i] = va_arg(ap, int);
  }
  va_end(ap);

  if (mem == NULL || *mem != NULL) return -1;
  *mem = (gpu_mem)malloc(sizeof(struct _gpu_memory_object));
  if (*mem == NULL) return -1;

  (*mem)->data_type = data_type;
  (*mem)->obj_type = DATA;
  (*mem)->reserved = true;
  (*mem)->distributed = true;
  (*mem)->consistent = false;

  switch (ndim) {
    case 4:
      return create_4d_tensor(*mem, data_type, args[0], args[1], args[2], args[3]);

    default:
      free(*mem);
      *mem = NULL;
      return -1;
  }
}

/* int create_buffer[DATA_GRADIENT](gpu_mem *, int, cudnnDataType, [int]) */
int create_buffer_data_gradient(gpu_mem *mem, int ndim, ...)
{
  int args[CUDNN_DIM_MAX];
  va_list ap;

  va_start(ap, ndim);
  cudnnDataType_t data_type = (cudnnDataType_t)va_arg(ap, int);
  for (int i = 0; i < ndim; i++) {
    args[i] = va_arg(ap, int);
  }
  va_end(ap);

  if (mem == NULL || *mem != NULL) return -1;
  *mem = (gpu_mem)malloc(sizeof(struct _gpu_memory_object));
  if (*mem == NULL) return -1;

  (*mem)->data_type = data_type;
  (*mem)->obj_type = DATA_GRADIENT;
  (*mem)->reserved = false;
  (*mem)->distributed = true;
  (*mem)->consistent = false;

  switch (ndim) {
    case 4:
      return create_4d_tensor(*mem, data_type, args[0], args[1], args[2], args[3]);

    default:
      free(*mem);
      *mem = NULL;
      return -1;
  }
}

static int create_4d_weight(
    gpu_mem mem, cudnnDataType_t data_type, int k, int c, int h, int w);

/* int create_buffer[WEIGHT](gpu_mem *, int, cudnnDataType, [int]) */
int create_buffer_weight(gpu_mem *mem, int ndim, ...)
{
  int args[CUDNN_DIM_MAX];
  va_list ap;

  va_start(ap, ndim);
  cudnnDataType_t data_type = (cudnnDataType_t)va_arg(ap, int);
  for (int i = 0; i < ndim; i++) {
    args[i] = va_arg(ap, int);
  }
  va_end(ap);

  if (mem == NULL || *mem != NULL) return -1;
  *mem = (gpu_mem)malloc(sizeof(struct _gpu_memory_object));
  if (*mem == NULL) return -1;

  (*mem)->data_type = data_type;
  (*mem)->obj_type = WEIGHT;
  (*mem)->reserved = true;
  (*mem)->distributed = false;
  (*mem)->consistent = true;

  switch (ndim) {
    case 4:
      return create_4d_weight(*mem, data_type, args[0], args[1], args[2], args[3]);

    default:
      free(*mem);
      *mem = NULL;
      return -1;
  }
}

/* int create_buffer[WEIGHT_GRADIENT](gpu_mem *, int, cudnnDataType, [int]) */
int create_buffer_weight_gradient(gpu_mem *mem, int ndim, ...)
{
  int args[CUDNN_DIM_MAX];
  va_list ap;

  va_start(ap, ndim);
  cudnnDataType_t data_type = (cudnnDataType_t)va_arg(ap, int);
  for (int i = 0; i < ndim; i++) {
    args[i] = va_arg(ap, int);
  }
  va_end(ap);

  if (mem == NULL || *mem != NULL) return -1;
  *mem = (gpu_mem)malloc(sizeof(struct _gpu_memory_object));
  if (*mem == NULL) return -1;

  (*mem)->data_type = data_type;
  (*mem)->obj_type = WEIGHT_GRADIENT;
  (*mem)->reserved = false;
  (*mem)->distributed = false;
  (*mem)->consistent = false;

  switch (ndim) {
    case 4:
      return create_4d_weight(*mem, data_type, args[0], args[1], args[2], args[3]);

    default:
      free(*mem);
      *mem = NULL;
      return -1;
  }
}

static int create_rawspace(gpu_mem mem, size_t size_in_bytes);

/* int create_buffer[WORK_SPACE](gpu_mem *, int, size_t) */
int create_buffer_work_space(gpu_mem *mem, int ndim, ...)
{
  va_list ap;

  va_start(ap, ndim);
  size_t size_in_bytes = va_arg(ap, size_t);
  va_end(ap);

  if (mem == NULL || *mem != NULL) return -1;
  *mem = (gpu_mem)malloc(sizeof(struct _gpu_memory_object));
  if (*mem == NULL) return -1;

  (*mem)->data_type = 0;
  (*mem)->obj_type = WORK_SPACE;
  (*mem)->reserved = false;
  (*mem)->distributed = false;
  (*mem)->consistent = false;

  return create_rawspace(*mem, size_in_bytes);
}

/* int create_buffer[RESERVE_SPACE](gpu_mem *, int, size_t) */
int create_buffer_reserve_space(gpu_mem *mem, int ndim, ...)
{
  va_list ap;

  va_start(ap, ndim);
  size_t size_in_bytes = va_arg(ap, size_t);
  va_end(ap);

  if (mem == NULL || *mem != NULL) return -1;
  *mem = (gpu_mem)malloc(sizeof(struct _gpu_memory_object));
  if (*mem == NULL) return -1;

  (*mem)->data_type = 0;
  (*mem)->obj_type = RESERVE_SPACE;
  (*mem)->reserved = true;
  (*mem)->distributed = false;
  (*mem)->consistent = false;

  return create_rawspace(*mem, size_in_bytes);
}

int destroy_buffer(gpu_mem mem)
{
  if (mem == NULL) return -1;

  if (mem->filter_desc) {
    chkCUDNN(cudnnDestroyFilterDescriptor(mem->filter_desc));
  }

  for (int dev = 0; dev < num_devices; dev++) {
    if (mem->tensor_desc[dev]) {
      chkCUDNN(cudnnDestroyTensorDescriptor(mem->tensor_desc[dev]));
    }

    if (mem->dev_ptr[dev]) {
      chkCUDA(cudaSetDevice(dev));
      chkCUDA(cudaFree(mem->dev_ptr[dev]));
    }
  }

  free(mem);

  return 0;
}

int create_4d_tensor(
    gpu_mem mem, cudnnDataType_t data_type, int n, int c, int h, int w)
{
  mem->ndim = 4;
  mem->dim[0] = n;
  mem->dim[1] = c;
  mem->dim[2] = h;
  mem->dim[3] = w;

  mem->filter_desc = NULL;

  for (int dev = 0; dev < num_devices; dev++) {
    chkCUDNN(cudnnCreateTensorDescriptor(&mem->tensor_desc[dev]));

    int n_dev = distribute(n, dev);

    chkCUDNN(cudnnSetTensor4dDescriptor(
          mem->tensor_desc[dev], CUDNN_TENSOR_NCHW,
          data_type, n_dev, c, h, w));

    mem->size_in_bytes[dev] = size_of_cudnn_data_type[data_type] * n_dev * c * h * w;
#if ON_DEMAND_ALLOCATION
    mem->dev_ptr[dev] = NULL;
#else
    chkCUDA(cudaSetDevice(dev));
    chkCUDA(cudaMalloc(&mem->dev_ptr[dev], mem->size_in_bytes[dev]));
#endif
  }

  return 0;
}

int create_4d_weight(
    gpu_mem mem, cudnnDataType_t data_type, int k, int c, int h, int w)
{
  mem->ndim = 4;
  mem->dim[0] = k;
  mem->dim[1] = c;
  mem->dim[2] = h;
  mem->dim[3] = w;

  chkCUDNN(cudnnCreateFilterDescriptor(&mem->filter_desc));

  chkCUDNN(cudnnSetFilter4dDescriptor(
        mem->filter_desc, data_type,
        CUDNN_TENSOR_NCHW, k, c, h, w));

  size_t size_in_bytes = size_of_cudnn_data_type[data_type] * k * c * h * w;

  for (int dev = 0; dev < num_devices; dev++) {
    chkCUDNN(cudnnCreateTensorDescriptor(&mem->tensor_desc[dev]));

    chkCUDNN(cudnnSetTensor4dDescriptor(
          mem->tensor_desc[dev], CUDNN_TENSOR_NCHW,
          data_type, k, c, h, w));

    mem->size_in_bytes[dev] = size_in_bytes;
#if ON_DEMAND_ALLOCATION
    mem->dev_ptr[dev] = NULL;
#else
    chkCUDA(cudaSetDevice(dev));
    chkCUDA(cudaMalloc(&mem->dev_ptr[dev], mem->size_in_bytes[dev]));
#endif
  }

  return 0;
}

int create_rawspace(gpu_mem mem, size_t size_in_bytes)
{
  mem->ndim = 1;
  mem->dim[0] = size_in_bytes;

  mem->filter_desc = NULL;

  for (int dev = 0; dev < num_devices; dev++) {
    mem->tensor_desc[dev] = NULL;

    mem->size_in_bytes[dev] = size_in_bytes;
#if ON_DEMAND_ALLOCATION
    mem->dev_ptr[dev] = NULL;
#else
    chkCUDA(cudaSetDevice(dev));
    chkCUDA(cudaMalloc(&mem->dev_ptr[dev], mem->size_in_bytes[dev]));
#endif
  }

  return 0;
}

////////////////////////////////////////////////////////////
// Memory Transfer API
////////////////////////////////////////////////////////////

int write_buffer(const gpu_mem dst, const void *src, bool synch)
{
  const char *host = (const char *)src;

  if (dst->distributed) {
    for (int dev = 0; dev < num_devices; dev++) {
      chkCUDA(cudaSetDevice(dev));

      chkCUDA(cudaMemcpyAsync(
            dst->dev_ptr[dev], host, dst->size_in_bytes[dev],
            cudaMemcpyHostToDevice, memory_stream[dev]));

      host += dst->size_in_bytes[dev];
    }
    if (synch) {
      for (int dev = 0; dev < num_devices; dev++) {
        chkCUDA(cudaStreamSynchronize(memory_stream[dev]));
      }
      // MPI_Barrier()
    }

    return 0;
  }
  else if (dst->consistent) {
    for (int dev = 0; dev < num_devices; dev++) {
      chkCUDA(cudaSetDevice(dev));

      chkCUDA(cudaMemcpyAsync(
            dst->dev_ptr[dev], host, dst->size_in_bytes[dev],
            cudaMemcpyHostToDevice, memory_stream[dev]));
    }
    if (synch) {
      for (int dev = 0; dev < num_devices; dev++) {
        chkCUDA(cudaStreamSynchronize(memory_stream[dev]));
      }
      // MPI_Barrier()
    }

    return 0;
  }

  return -1;
}

int read_buffer(void *dst, const gpu_mem src, bool synch)
{
  char *host = (char *)dst;

  if (src->distributed) {
    for (int dev = 0; dev < num_devices; dev++) {
      chkCUDA(cudaMemcpyAsync(
            host, src->dev_ptr[dev], src->size_in_bytes[dev],
            cudaMemcpyDeviceToHost, memory_stream[dev]));

      host += src->size_in_bytes[dev];
    }
    if (synch) {
      for (int dev = 0; dev < num_devices; dev++) {
        chkCUDA(cudaStreamSynchronize(memory_stream[dev]));
      }
      // MPI_Barrier()
    }

    return 0;
  }
  else if (src->consistent) {
    for (int dev = 0; dev < 1; dev++) {
      chkCUDA(cudaMemcpyAsync(
            host, src->dev_ptr[dev], src->size_in_bytes[dev],
            cudaMemcpyDeviceToHost, memory_stream[dev]));
    }
    if (synch) {
      for (int dev = 0; dev < 1; dev++) {
        chkCUDA(cudaStreamSynchronize(memory_stream[dev]));
      }
      // MPI_Barrier()
    }

    return 0;
  }

  return -1;
}

static bool is_mem_equivalent(const gpu_mem a, const gpu_mem b);

int copy_buffer(const gpu_mem dst, const gpu_mem src, bool synch)
{
  if (!is_mem_equivalent(dst, src)) return -1;

  if (src->distributed) {
    for (int dev = 0; dev < num_devices; dev++) {
      chkCUDA(cudaMemcpyAsync(
            dst->dev_ptr[dev], src->dev_ptr[dev], src->size_in_bytes[dev],
            cudaMemcpyDeviceToDevice, memory_stream[dev]));
    }
    if (synch) {
      for (int dev = 0; dev < num_devices; dev++) {
        chkCUDA(cudaStreamSynchronize(memory_stream[dev]));
      }
      // MPI_Barrier()
    }
  }
  else {
    for (int dev = 0; dev < num_devices; dev++) {
      chkCUDA(cudaMemcpyAsync(
            dst->dev_ptr[dev], src->dev_ptr[dev], src->size_in_bytes[dev],
            cudaMemcpyDeviceToDevice, memory_stream[dev]));
    }
    if (synch) {
      for (int dev = 0; dev < num_devices; dev++) {
        chkCUDA(cudaStreamSynchronize(memory_stream[dev]));
      }
      // MPI_Barrier()
    }
  }

  return 0;
}

bool is_mem_equivalent(const gpu_mem a, const gpu_mem b)
{
  if (a->ndim != b->ndim) return false;

  for (int i = 0; i < a->ndim; i++) {
    if (a->dim[i] != b->dim[i]) return false;
  }

  if (a->reserved != b->reserved) return false;
  if (a->distributed != b->distributed) return false;
  if (a->consistent != b->consistent) return false;

  return true;
}

int all_reduce_buffer(const gpu_mem mem, bool synch)
{
  ncclGroupStart();

  for (int dev = 0; dev < num_devices; dev++) {
    // FIXME: only float type works for a while...
    ncclAllReduce(
        (const void *)mem->dev_ptr[dev], (void *)mem->dev_ptr[dev],
        mem->size_in_bytes[dev] / sizeof(float),
        ncclFloat, ncclSum, nccl_comm[dev], memory_stream[dev]);
  }

  ncclGroupEnd();

  if (synch) {
    for (int dev = 0; dev < num_devices; dev++) {
      chkCUDA(cudaStreamSynchronize(memory_stream[dev]));
    }
    // MPI_Barrier()
  }

  mem->consistent = true;

  return 0;
}
