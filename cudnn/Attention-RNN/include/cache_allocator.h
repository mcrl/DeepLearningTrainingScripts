#ifndef __CACHE_ALLOCATOR_H__
#define __CACHE_ALLOCATOR_H__

#include <cuda_runtime.h>
void cacher_init(size_t size);
void cacher_clear();
cudaError_t cacher_alloc(void **m, size_t bsize);
cudaError_t cacher_free(void *);

#endif //__CACHE_ALLOCATOR_H__
