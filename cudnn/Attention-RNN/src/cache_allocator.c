#include <assert.h>
#include "gpu.h"
#include "utils.h"

static void *buf;
static size_t total;
static size_t used;

//#define __SER__
void cacher_init(size_t size)
{
#ifndef __SER__
	chkCUDA(cudaMalloc(&buf, size));
	printf("cacher init with %lu\n", size);
	total = size;
	used = 0;
#endif
}

void cacher_clear()
{
#ifndef __SER__
	used = 0;
#endif
}


cudaError_t cacher_alloc(void **m, size_t size)
{
#ifndef __SER__
	*m = (buf + used);
	size = (size + 255) / 256 * 256;
	used += size;
	if (used >= total)
		assert(0);
	return cudaSuccess;
#else
	return cudaMalloc(m, size);
#endif
}

cudaError_t cacher_free(void *m)
{
#ifndef __SER__
	return cudaSuccess;
#else
	return cudaFree(m);
#endif
}
