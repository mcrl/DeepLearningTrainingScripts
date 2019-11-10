#include <assert.h>
#include <stdlib.h>

#include "rstack.h"
#include "gpu.h"
#include "cache_allocator.h"
struct rstack {
	void **reserves;
	size_t *reserve_sizes;
	int cnt;
	int nelem;
	int max_cnt;
};


struct rstack* rstack_create(int init_size)
{
	struct rstack *rstack = malloc(sizeof(struct rstack));
	rstack->nelem = init_size;
	rstack->cnt = 0;
	rstack->max_cnt = 0;
	rstack->reserves = malloc(sizeof(void*) * init_size);
	rstack->reserve_sizes = malloc(sizeof(size_t) * init_size);
	return rstack;
}

void rstack_free(struct rstack *rstack)
{
	free(rstack->reserves);
	free(rstack->reserve_sizes);
}


void rstack_push(struct rstack *rstack, size_t size)
{
	if (rstack->cnt == rstack->nelem) {
		rstack->nelem *= 2;
		rstack->reserves = realloc(rstack->reserves,
								sizeof(void*) * rstack->nelem);
		rstack->reserve_sizes = realloc(rstack->reserve_sizes,
								sizeof(size_t) * rstack->nelem);
	}

	chkCUDA(cacher_alloc(&rstack->reserves[rstack->cnt], size));
	rstack->reserve_sizes[rstack->cnt] = size;
	rstack->cnt++;

	if (rstack->cnt > rstack->max_cnt) {
		rstack->max_cnt = rstack->cnt;
	}
}

void rstack_pop(struct rstack *rstack)
{
	assert(rstack->cnt > 0);
	rstack->cnt--;
}

size_t rstack_top_size(struct rstack *rstack)
{
	assert(rstack->cnt > 0);
	return rstack->reserve_sizes[(rstack->cnt)-1];
}

void* rstack_top(struct rstack *rstack)
{
	assert(rstack->cnt > 0);
	return rstack->reserves[(rstack->cnt)-1];
}

int rstack_size(struct rstack * rstack)
{
	return rstack->cnt;
}

void rstack_clear(struct rstack *rstack)
{
	//assert(rstack->cnt == 0);
	for (int i = 0; i < rstack->max_cnt; i++) {
		chkCUDA(cacher_free(rstack->reserves[i]));
		rstack->reserves[i] = NULL;
		rstack->reserve_sizes[i] = 0;
	}
	rstack->cnt = 0;
	rstack->max_cnt = 0;
}
