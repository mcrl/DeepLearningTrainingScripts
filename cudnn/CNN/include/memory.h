#ifndef _MEMORY_H_
#define _MEMORY_H_

#include <stdlib.h>
#include <stdbool.h>
#include <stdarg.h>

#include <cuda.h>
#include <cudnn.h>

#include "list.h"

#define MAX_NDEV 4

////////////////////////////////////////////////////////////
// Abstract Device Memory Object
////////////////////////////////////////////////////////////

typedef enum {
  DATA,               // reserved distributed input/output
  DATA_GRADIENT,      // reusable distributed input/output
  WEIGHT,             // reserved shared input
  WEIGHT_GRADIENT,    // reusable shared output
  BN_PARAM,           // reserved unique input
  BN_PARAM_GRADIENT,  // reusable unique output
  WORK_SPACE,         // reusable private output
  RESERVE_SPACE,      // reusable private input/output
  NUM_OBJECT_TYPE
} gpu_memory_object_t;

struct _gpu_memory_object {
  list_t *parent;
  iterator_t iterator;

  void *dev_ptr[MAX_NDEV];
  size_t size_in_bytes[MAX_NDEV];

  int ndim;
  size_t dim[CUDNN_DIM_MAX];

  cudnnDataType_t data_type;
  cudnnTensorDescriptor_t tensor_desc[MAX_NDEV];
  cudnnFilterDescriptor_t filter_desc;

  gpu_memory_object_t obj_type;

  bool allocated;
  bool reserved;
  bool distributed;
  bool consistent;
};

typedef struct _gpu_memory_object *gpu_mem;

#define is_data(mem) ( (mem)->obj_type == DATA )

#define is_data_grad(mem) ( (mem)->obj_type == DATA_GRADIENT )

#define is_data_or_data_grad(mem) ( is_data(mem) || is_data_grad(mem) )

#define is_weight(mem) ( (mem)->obj_type == WEIGHT )

#define is_weight_grad(mem) ( (mem)->obj_type == WEIGHT_GRADIENT )

#define is_bn_param(mem) ( (mem)->obj_type == BN_PARAM )

#define is_bn_param_grad(mem) ( (mem)->obj_type == BN_PARAM_GRADIENT )

#define is_weight_or_param(mem) ( is_weight(mem) || is_bn_param(mem) )

#define is_weight_grad_or_param_grad(mem) ( is_weight_grad(mem) || is_bn_param_grad(mem) )

#define is_work_space(mem) ( (mem)->obj_type == WORK_SPACE )

#define is_reserve_space(mem) ( (mem)->obj_type == RESERVE_SPACE )

size_t data_type_size(gpu_mem mem);

size_t logical_buffer_size(gpu_mem mem);

////////////////////////////////////////////////////////////
// Memory Object Management API
////////////////////////////////////////////////////////////

int __init_object_manager(void);

int __finalize_object_manager(void);

int create_buffer_data(
    gpu_mem *mem, cudnnDataType_t data_type, int ndim, ...);

int create_buffer_data_gradient(
    gpu_mem *mem, cudnnDataType_t data_type, int ndim, ...);

int create_buffer_weight(
    gpu_mem *mem, cudnnDataType_t data_type, int ndim, ...);

int create_buffer_weight_gradient(
    gpu_mem *mem, cudnnDataType_t data_type, int ndim, ...);

int create_buffer_bn_param(
    gpu_mem *mem, cudnnDataType_t data_type,
    cudnnBatchNormMode_t mode, int ndim, ...);

int create_buffer_bn_param_gradient(
    gpu_mem *mem, cudnnDataType_t data_type,
    cudnnBatchNormMode_t mode, int ndim, ...);

int create_buffer_work_space(gpu_mem *mem, size_t size_in_bytes);

int create_buffer_reserve_space(gpu_mem *mem, size_t size_in_bytes);

int destroy_buffer(gpu_mem mem);

////////////////////////////////////////////////////////////
// Device Memory Management API
////////////////////////////////////////////////////////////

int bind_buffer1(gpu_mem trg);

int bind_buffer2(gpu_mem trg, gpu_mem inc);

int bind_buffer3(gpu_mem trg, gpu_mem inc, gpu_mem exc, int j);

int alloc_buffer(gpu_mem mem);

int alloc_buffer_by_type(gpu_memory_object_t obj_type);

int free_buffer(gpu_mem mem);

int free_buffer_by_type(gpu_memory_object_t obj_type);

////////////////////////////////////////////////////////////
// Memory Transfer API
////////////////////////////////////////////////////////////

int write_buffer(const gpu_mem dst, const void *src, bool synch);

int read_buffer(void *dst, const gpu_mem src, bool synch);

int copy_buffer(const gpu_mem dst, const gpu_mem src, bool synch);

int all_reduce_buffer(const gpu_mem mem, bool synch);

#endif // _MEMORY_H_
