#include <stdlib.h>
#include <stdbool.h>

#include <cuda.h>
#include <cudnn.h>
#include <nccl.h>

#include "memory.h"
#include "utils.h"
#include "list.h"

int node_id = 0;
int num_nodes = 1;
int num_devices = 1;
ncclUniqueId nccl_id;

cudaStream_t memory_stream[MAX_NDEV];
ncclComm_t nccl_comm[MAX_NDEV];

struct _gpu_memory_object_group {
  iterator_t iterator;
  list_t mem_list;
};

typedef struct _gpu_memory_object_group *gpu_mem_group;

list_t mem_group_list[NUM_OBJECT_TYPE];

static const size_t size_of_cudnn_data_type[] = { 4, 8, 2, 1, 4, 4, 1, 4, 32 };

#define distribute(n, dev) ( ((n) / num_devices) + ((dev) < (n) % num_devices) )

#define resolve_params(type, args, argc) \
do {\
  va_list ap;\
  va_start(ap, argc);\
  for (int i = 0; i < argc; i++) {\
    args[i] = va_arg(ap, type);\
  }\
  va_end(ap);\
} while (0)

#define check(expr) if (!(expr)) goto error

////////////////////////////////////////////////////////////
// Abstract Device Memory Object
////////////////////////////////////////////////////////////

size_t data_type_size(gpu_mem mem)
{
  return size_of_cudnn_data_type[(int)mem->data_type];
}

size_t logical_buffer_size(gpu_mem mem)
{
  if (mem->distributed) {
    size_t total_size = 0;
    for (int dev = 0; dev < num_devices; dev++) {
      total_size += mem->size_in_bytes[dev];
    }
    return total_size;
  }

  return mem->size_in_bytes[0];
}

////////////////////////////////////////////////////////////
// Memory Object Management API
////////////////////////////////////////////////////////////

int __init_object_manager()
{
  static bool initialized = false;

  check(!initialized);

  for (int type = 0; type < NUM_OBJECT_TYPE; type++) {
    list_init(&mem_group_list[type]);
  }

  ncclGroupStart();

  for (int dev = 0; dev < num_devices; dev++) {
    chkCUDA(cudaSetDevice(dev));
    chkCUDA(cudaStreamCreate(&memory_stream[dev]));
    ncclCommInitRank(&nccl_comm[dev], num_nodes * num_devices, nccl_id, node_id * num_devices + dev);
  }

  ncclGroupEnd();

  initialized = true;

  return 0;

error:
  return -1;
}

int __finalize_object_manager()
{
  /*
  for (int type = 0; type < NUM_OBJECT_TYPE; type++) {
    list_iter(&mem_group_list[type]) {
      gpu_mem_group group = list_data(struct _gpu_memory_object_group, iterator);
      list_iter(&group->mem_list) {
        gpu_mem mem = list_data(struct _gpu_memory_object, iterator);
        destroy_buffer(mem);
      }
    }
  }
  */

  for (int dev = 0; dev < num_devices; dev++) {
    chkCUDA(cudaSetDevice(dev));
    chkCUDA(cudaStreamDestroy(memory_stream[dev]));
    ncclCommDestroy(nccl_comm[dev]);
  }

  return 0;
}

static void assign_flags_from_object_type(gpu_mem mem);

static int create_4d_tensor(
    gpu_mem mem, cudnnDataType_t data_type, int n, int c, int h, int w);

int create_buffer_data(
    gpu_mem *mem, cudnnDataType_t data_type, int ndim, ...)
{
  int args[CUDNN_DIM_MAX];

  resolve_params(int, args, ndim);

  check(mem);
  *mem = (gpu_mem)malloc(sizeof(struct _gpu_memory_object));
  check(*mem);

  (*mem)->parent = NULL;
  (*mem)->data_type = data_type;
  (*mem)->obj_type = DATA;
  (*mem)->allocated = false;

  assign_flags_from_object_type(*mem);

  switch (ndim) {
    case 4:
      return create_4d_tensor(*mem, data_type, args[0], args[1], args[2], args[3]);

    default:
      free(*mem);
      *mem = NULL;
      goto error;
  }

error:
  return -1;
}

int create_buffer_data_gradient(
    gpu_mem *mem, cudnnDataType_t data_type, int ndim, ...)
{
  int args[CUDNN_DIM_MAX];

  resolve_params(int, args, ndim);

  check(mem);
  *mem = (gpu_mem)malloc(sizeof(struct _gpu_memory_object));
  check(*mem);

  (*mem)->parent = NULL;
  (*mem)->data_type = data_type;
  (*mem)->obj_type = DATA_GRADIENT;
  (*mem)->allocated = false;

  assign_flags_from_object_type(*mem);

  switch (ndim) {
    case 4:
      return create_4d_tensor(*mem, data_type, args[0], args[1], args[2], args[3]);

    default:
      free(*mem);
      *mem = NULL;
      goto error;
  }

error:
  return -1;
}

static int create_4d_weight(
    gpu_mem mem, cudnnDataType_t data_type, int k, int c, int h, int w);

int create_buffer_weight(
    gpu_mem *mem, cudnnDataType_t data_type, int ndim, ...)
{
  int args[CUDNN_DIM_MAX];

  resolve_params(int, args, ndim);

  check(mem);
  *mem = (gpu_mem)malloc(sizeof(struct _gpu_memory_object));
  check(*mem);

  (*mem)->parent = NULL;
  (*mem)->data_type = data_type;
  (*mem)->obj_type = WEIGHT;
  (*mem)->allocated = false;

  assign_flags_from_object_type(*mem);

  switch (ndim) {
    case 4:
      return create_4d_weight(*mem, data_type, args[0], args[1], args[2], args[3]);

    default:
      free(*mem);
      *mem = NULL;
      goto error;
  }

error:
  return -1;
}

int create_buffer_weight_gradient(
    gpu_mem *mem, cudnnDataType_t data_type, int ndim, ...)
{
  int args[CUDNN_DIM_MAX];

  resolve_params(int, args, ndim);

  check(mem);
  *mem = (gpu_mem)malloc(sizeof(struct _gpu_memory_object));
  check(*mem);

  (*mem)->parent = NULL;
  (*mem)->data_type = data_type;
  (*mem)->obj_type = WEIGHT_GRADIENT;
  (*mem)->allocated = false;

  assign_flags_from_object_type(*mem);

  switch (ndim) {
    case 4:
      return create_4d_weight(*mem, data_type, args[0], args[1], args[2], args[3]);

    default:
      free(*mem);
      *mem = NULL;
      goto error;
  }

error:
  return -1;
}

static void derive_bn_shape(
    cudnnBatchNormMode_t mode, int shape[], int ndim);

int create_buffer_bn_param(
    gpu_mem *mem, cudnnDataType_t data_type,
    cudnnBatchNormMode_t mode, int ndim, ...)
{
  int args[CUDNN_DIM_MAX];

  resolve_params(int, args, ndim);

  check(mem);
  *mem = (gpu_mem)malloc(sizeof(struct _gpu_memory_object));
  check(*mem);

  (*mem)->parent = NULL;
  (*mem)->data_type = data_type;
  (*mem)->obj_type = BN_PARAM;
  (*mem)->allocated = false;

  assign_flags_from_object_type(*mem);

  derive_bn_shape(mode, args, ndim);

  switch (ndim) {
    case 4:
      return create_4d_weight(*mem, data_type, args[0], args[1], args[2], args[3]);

    default:
      free(*mem);
      *mem = NULL;
      goto error;
  }

error:
  return -1;
}

int create_buffer_bn_param_gradient(
    gpu_mem *mem, cudnnDataType_t data_type,
    cudnnBatchNormMode_t mode, int ndim, ...)
{
  int args[CUDNN_DIM_MAX];

  resolve_params(int, args, ndim);

  check(mem);
  *mem = (gpu_mem)malloc(sizeof(struct _gpu_memory_object));
  check(*mem);

  (*mem)->parent = NULL;
  (*mem)->data_type = data_type;
  (*mem)->obj_type = BN_PARAM_GRADIENT;
  (*mem)->allocated = false;

  assign_flags_from_object_type(*mem);

  derive_bn_shape(mode, args, ndim);

  switch (ndim) {
    case 4:
      return create_4d_weight(*mem, data_type, args[0], args[1], args[2], args[3]);

    default:
      free(*mem);
      *mem = NULL;
      goto error;
  }

error:
  return -1;
}

static int create_rawspace(gpu_mem mem, size_t size_in_bytes);

int create_buffer_work_space(gpu_mem *mem, size_t size_in_bytes)
{
  check(mem);
  *mem = (gpu_mem)malloc(sizeof(struct _gpu_memory_object));
  check(*mem);

  (*mem)->parent = NULL;
  (*mem)->data_type = 0;
  (*mem)->obj_type = WORK_SPACE;
  (*mem)->allocated = false;

  assign_flags_from_object_type(*mem);

  return create_rawspace(*mem, size_in_bytes);

error:
  return -1;
}

int create_buffer_reserve_space(gpu_mem *mem, size_t size_in_bytes)
{
  check(mem);
  *mem = (gpu_mem)malloc(sizeof(struct _gpu_memory_object));
  check(*mem);

  (*mem)->parent = NULL;
  (*mem)->data_type = 0;
  (*mem)->obj_type = RESERVE_SPACE;
  (*mem)->allocated = false;

  assign_flags_from_object_type(*mem);

  return create_rawspace(*mem, size_in_bytes);

error:
  return -1;
}

// FIXME: deprecated
int destroy_buffer(gpu_mem mem)
{
  if (!mem) goto error;

  if (mem->filter_desc) {
    chkCUDNN(cudnnDestroyFilterDescriptor(mem->filter_desc));
  }

  for (int dev = 0; dev < num_devices; dev++) {
    if (mem->tensor_desc[dev]) {
      chkCUDNN(cudnnDestroyTensorDescriptor(mem->tensor_desc[dev]));
    }

    if (mem->dev_ptr[dev] && mem->allocated) {
      chkCUDA(cudaSetDevice(dev));
      chkCUDA(cudaFree(mem->dev_ptr[dev]));
    }
  }

  if (mem->parent) {
    list_erase(mem->parent, &mem->iterator);
  }

  free(mem);

  return 0;

error:
  return -1;
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
    mem->dev_ptr[dev] = NULL;
  }
  mem->allocated = false;

  return 0;

error:
  return -1;
}

static int alloc_buffer_internal(gpu_mem mem);

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
    mem->dev_ptr[dev] = NULL;
  }
  mem->allocated = false;

  check(alloc_buffer_internal(mem) == 0); // FIXME: add to list

  return 0;

error:
  return -1;
}

static int create_group(gpu_mem mem);

int create_rawspace(gpu_mem mem, size_t size_in_bytes)
{
  mem->ndim = 1;
  mem->dim[0] = size_in_bytes;

  mem->filter_desc = NULL;

  for (int dev = 0; dev < num_devices; dev++) {
    mem->tensor_desc[dev] = NULL;
    mem->size_in_bytes[dev] = size_in_bytes;
    mem->dev_ptr[dev] = NULL;
  }
  mem->allocated = false;

  static bool is_first = true;

  if (is_first) {
    create_group(mem);
    is_first = false;
  }
  else {
    list_t *l = &mem_group_list[mem->obj_type];

    gpu_mem_group group =
      list_first(l, struct _gpu_memory_object_group, iterator);

    mem->parent = &group->mem_list;
    list_push_back(mem->parent, &mem->iterator);
  }

  return 0;

error:
  return -1;
}

void derive_bn_shape(cudnnBatchNormMode_t mode, int shape[], int ndim)
{
  shape[0] = 1;

  if (mode != CUDNN_BATCHNORM_PER_ACTIVATION) {
    for (int i = 2; i < ndim; i++) {
      shape[i] = 1;
    }
  }
}

void assign_flags_from_object_type(gpu_mem mem)
{
  switch (mem->obj_type) {
    case DATA:
      mem->reserved = true;
      mem->distributed = true;
      mem->consistent = false;
      break;

    case DATA_GRADIENT:
      mem->reserved = false;
      mem->distributed = true;
      mem->consistent = false;
      break;

    case WEIGHT:
      mem->reserved = true;
      mem->distributed = false;
      mem->consistent = true;
      break;

    case WEIGHT_GRADIENT:
      mem->reserved = false;
      mem->distributed = false;
      mem->consistent = false;
      break;

    case BN_PARAM:
      mem->reserved = true;
      mem->distributed = false;
      mem->consistent = false;
      break;

    case BN_PARAM_GRADIENT:
      mem->reserved = false;
      mem->distributed = false;
      mem->consistent = false;
      break;

    case WORK_SPACE:
      mem->reserved = false;
      mem->distributed = false;
      mem->consistent = false;
      break;

    case RESERVE_SPACE:
      mem->reserved = true;
      mem->distributed = false;
      mem->consistent = false;
      break;
  }
}

////////////////////////////////////////////////////////////
// Device Memory Management API
////////////////////////////////////////////////////////////

int bind_buffer1(gpu_mem trg)
{
  LOG(begin);
  check(!trg->parent);
  LOG(passed_precondition);

  check(create_group(trg) == 0);

  LOG(end);
  return 0;

error:
  LOG(error);
  return -1;
}

int bind_buffer2(gpu_mem trg, gpu_mem inc)
{
  LOG(begin);
  check(!trg->parent);
  check(trg->obj_type == inc->obj_type);
  LOG(passed_precondition);

  if (!inc->parent) {
    LOG(inc_is_not_allocated);
    check(create_group(inc) == 0);
  }

  LOG(add_trg_to_group);
  trg->parent = inc->parent;
  list_push_back(trg->parent, &trg->iterator);

  LOG(end);
  return 0;

error:
  LOG(error);
  return -1;
}

int bind_buffer3(gpu_mem trg, gpu_mem inc, gpu_mem exc, int j)
{
  LOG(begin);
  check(!trg->parent);
  check(trg->obj_type == inc->obj_type);

  check(!inc->parent);
  check(inc->obj_type == exc->obj_type);
  LOG(passed_precondition);

  gpu_mem_group dst_group = NULL;

  list_t *l = &mem_group_list[trg->obj_type];

  list_iter(l) {
    gpu_mem_group group =
      list_data(struct _gpu_memory_object_group, iterator);

    if (&group->mem_list != exc->parent) {
      if (j-- == 0) {
        LOG(found_dst_group);
        dst_group = group;
        LOG(add_inc_to_group);
        inc->parent = &dst_group->mem_list;
        list_push_back(inc->parent, &inc->iterator);
        break;
      }
    }
  }

  if (!dst_group) {
    LOG(not_found_dst_group);
    check(create_group(inc) == 0);
  }

  LOG(add_trg_to_group);
  trg->parent = inc->parent;
  list_push_back(trg->parent, &trg->iterator);

  LOG(end);
  return 0;

error:
  LOG(error);
  return -1;
}

int create_group(gpu_mem mem)
{
  LOG(begin);
  check(!mem->parent);
  LOG(passed_precondition);

  LOG(malloc_new_group);
  gpu_mem_group new_group =
    (gpu_mem_group)malloc(sizeof(struct _gpu_memory_object_group));
  LOG(insert_new_group_to_list);
  list_push_back(&mem_group_list[mem->obj_type], &new_group->iterator);
  list_init(&new_group->mem_list);

  mem->parent = &new_group->mem_list;
  LOG(add_mem_to_group);
  list_push_back(mem->parent, &mem->iterator);

  LOG(end);
  return 0;

error:
  LOG(error);
  return -1;
}

static gpu_mem find_largest_buffer(list_t *mem_list);

static int alloc_buffer_group(list_t *mem_list)
{
  LOG(begin);
  gpu_mem max_mem = find_largest_buffer(mem_list);
  LOG(found_largest_buffer);

  check(alloc_buffer_internal(max_mem) == 0);

  list_iter(mem_list) {
    gpu_mem mem = list_data(struct _gpu_memory_object, iterator);

    LOG(copy_pointer);
    for (int dev = 0; dev < num_devices; dev++) {
      mem->dev_ptr[dev] = max_mem->dev_ptr[dev];
    }
    mem->allocated = true;
  }

  LOG(end);
  return 0;

error:
  LOG(error);
  return -1;
}

static int free_buffer_internal(gpu_mem mem);

static int free_buffer_group(list_t *mem_list)
{
  list_iter(mem_list) {
    gpu_mem mem = list_data(struct _gpu_memory_object, iterator);

    static bool is_first = true;

    if (is_first) {
      check(free_buffer_internal(mem) == 0);
      is_first = false;
    }

    for (int dev = 0; dev < num_devices; dev++) {
      mem->dev_ptr[dev] = NULL;
    }
    mem->allocated = false;
  }

  return 0;

error:
  return -1;
}

int alloc_buffer_internal(gpu_mem mem)
{
  LOG(begin);
  check(mem);
  check(!mem->allocated);
  LOG(passed_precondition);

  for (int dev = 0; dev < num_devices; dev++) {
    mem->dev_ptr[dev] = NULL;
    chkCUDA(cudaSetDevice(dev));
    chkCUDA(cudaMalloc(&mem->dev_ptr[dev], mem->size_in_bytes[dev]));
  }
  mem->allocated = true;

  LOG(end);
  return 0;

error:
  LOG(error);
  return -1;
}

int free_buffer_internal(gpu_mem mem)
{
  check(mem);
  check(mem->allocated);

  for (int dev = 0; dev < num_devices; dev++) {
    chkCUDA(cudaSetDevice(dev));
    chkCUDA(cudaFree(mem->dev_ptr[dev]));
    mem->dev_ptr[dev] = NULL;
  }
  mem->allocated = false;

  return 0;

error:
  return -1;
}

gpu_mem find_largest_buffer(list_t *mem_list)
{
  LOG(begin);
  gpu_mem max_mem = NULL;
  size_t max_size = 0;

  list_iter(mem_list) {
    gpu_mem mem = list_data(struct _gpu_memory_object, iterator);

    if (mem->size_in_bytes[0] > max_size) {
      LOG(update_max);
      max_mem = mem;
      max_size = mem->size_in_bytes[0];
    }
  }

  LOG(end);
  return max_mem;
}

int alloc_buffer(gpu_mem mem)
{
  LOG(begin);
  check(mem);

  if (!mem->allocated) {
    LOG(allocate);
    if (mem->parent) {
      check(alloc_buffer_group(mem->parent) == 0);
    }
    else {
      check(alloc_buffer_internal(mem) == 0);
    }
  }

  LOG(end);
  return 0;

error:
  LOG(error);
  return -1;
}

int alloc_buffer_by_type(gpu_memory_object_t obj_type)
{
  LOG(begin);
  list_t *l = &mem_group_list[obj_type];

  list_iter(l) {
    gpu_mem_group group =
      list_data(struct _gpu_memory_object_group, iterator);

    check(alloc_buffer_group(&group->mem_list) == 0);
  }

  LOG(end);
  return 0;

error:
  LOG(error);
  return -1;
}

int free_buffer(gpu_mem mem)
{
  LOG(begin);
  check(mem);

  if (mem->allocated) {
    LOG(release);
    if (mem->parent) {
      check(free_buffer_group(mem->parent) == 0);
    }
    else {
      check(free_buffer_internal(mem) == 0);
    }
  }

  LOG(end);
  return 0;

error:
  LOG(error);
  return -1;
}

int free_buffer_by_type(gpu_memory_object_t obj_type)
{
  LOG(begin);
  list_t *l = &mem_group_list[obj_type];

  list_iter(l) {
    gpu_mem_group group =
      list_data(struct _gpu_memory_object_group, iterator);

    check(free_buffer_group(&group->mem_list) == 0);
  }

  LOG(end);
  return 0;

error:
  LOG(error);
  return -1;
}

////////////////////////////////////////////////////////////
// Memory Transfer API
////////////////////////////////////////////////////////////

int write_buffer(const gpu_mem dst, const void *src, bool synch)
{
  if (!dst || !src) goto error;
  if (!dst->allocated) goto error;

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
  }
  else {
    goto error;
  }

  return 0;

error:
  return -1;
}

int read_buffer(void *dst, const gpu_mem src, bool synch)
{
  if (!src || !dst) goto error;
  if (!src->allocated) goto error;

  char *host = (char *)dst;

  if (src->distributed) {
    for (int dev = 0; dev < num_devices; dev++) {
      chkCUDA(cudaSetDevice(dev));

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
  }
  else if (src->consistent) {
    for (int dev = 0; dev < 1; dev++) {
      chkCUDA(cudaSetDevice(dev));

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
  }
  else {
    goto error;
  }

  return 0;

error:
  return -1;
}

static bool is_mem_equivalent(const gpu_mem a, const gpu_mem b);

int copy_buffer(const gpu_mem dst, const gpu_mem src, bool synch)
{
  if (!dst || !src) goto error;
  if (!dst->allocated || !src->allocated) goto error;
  if (!is_mem_equivalent(dst, src)) goto error;

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

error:
  return -1;
}

bool is_mem_equivalent(const gpu_mem a, const gpu_mem b)
{
  if (a->ndim != b->ndim) return false;

  for (int i = 0; i < a->ndim; i++) {
    if (a->dim[i] != b->dim[i]) return false;
  }

  return a->obj_type == b->obj_type;
}

int all_reduce_buffer(const gpu_mem mem, bool synch)
{
  if (!mem->allocated) return -1;

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

  return 0;
}
