#ifndef _MEMORY_H_
#define _MEMORY_H_

#include <stdlib.h>
#include <stdbool.h>
#include <stdarg.h>

#include <cuda.h>
#include <cudnn.h>

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
  RESERVE_SPACE       // reusable private input/output
} gpu_memory_object_t;

struct _gpu_memory_object {
  void *dev_ptr[MAX_NDEV];
  size_t size_in_bytes[MAX_NDEV];

  int ndim;
  size_t dim[CUDNN_DIM_MAX];

  cudnnTensorDescriptor_t tensor_desc[MAX_NDEV];
  cudnnFilterDescriptor_t filter_desc;

  cudnnDataType_t data_type;
  gpu_memory_object_t obj_type;

  bool reserved;
  bool distributed;
  bool consistent;
};

typedef struct _gpu_memory_object *gpu_mem;

size_t get_buffer_size(gpu_mem mem);

////////////////////////////////////////////////////////////
// Memory Object Management API
////////////////////////////////////////////////////////////

int __init_object_manager(void);

int __finalize_object_manager(void);

/* int create_buffer[DATA](gpu_mem *, int, cudnnDataType_t, [int]) */
int create_buffer_data(gpu_mem *mem, int ndim, ...);

/* int create_buffer[DATA_GRADIENT](gpu_mem *, int, cudnnDataType_t, [int]) */
int create_buffer_data_gradient(gpu_mem *mem, int ndim, ...);

/* int create_buffer[WEIGHT](gpu_mem *, int, cudnnDataType_t, [int]) */
int create_buffer_weight(gpu_mem *mem, int ndim, ...);

/* int create_buffer[WEIGHT_GRADIENT](gpu_mem *, int, cudnnDataType_t, [int]) */
int create_buffer_weight_gradient(gpu_mem *mem, int ndim, ...);

/* int create_buffer[BN_PARAM](gpu_mem *, int, cudnnDataType_t, cudnnBatchNormMode_t, [int]) */
int create_buffer_bn_param(gpu_mem *mem, int ndim, ...);

/* int create_buffer[BN_PARAM_GRADIENT](gpu_mem *, int, cudnnDataType_t, cudnnBatchNormMode_t, [int]) */
int create_buffer_bn_param_gradient(gpu_mem *mem, int ndim, ...);

/* int create_buffer[WORK_SPACE](gpu_mem *, int, size_t) */
int create_buffer_work_space(gpu_mem *mem, int ndim, ...);

/* int create_buffer[RESERVE_SPACE](gpu_mem *, int, size_t) */
int create_buffer_reserve_space(gpu_mem *mem, int ndim, ...);

typedef int (*create_buffer_t)(gpu_mem *, int, ...);

const create_buffer_t create_buffer[] = {
  create_buffer_data,
  create_buffer_data_gradient,
  create_buffer_weight,
  create_buffer_weight_gradient,
  create_buffer_bn_param,
  create_buffer_bn_param_gradient,
  create_buffer_work_space,
  create_buffer_reserve_space
};

int destroy_buffer(gpu_mem mem);

////////////////////////////////////////////////////////////
// Memory Transfer API
////////////////////////////////////////////////////////////

int write_buffer(const gpu_mem dst, const void *src, bool synch);

int read_buffer(void *dst, const gpu_mem src, bool synch);

int copy_buffer(const gpu_mem dst, const gpu_mem src, bool synch);

int all_reduce_buffer(const gpu_mem mem, bool synch);

#endif // _MEMORY_H_
