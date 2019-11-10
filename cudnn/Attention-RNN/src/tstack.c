#include <assert.h>
#include <stdlib.h>

#include "tstack.h"
#include "gpu.h"
struct tstack {
	tensor_t **tensors;
	int cnt;
	int nelem;
	int max_cnt;
};


struct tstack* tstack_create(int init_size)
{
	struct tstack *tstack = malloc(sizeof(struct tstack));
	tstack->nelem = init_size;
	tstack->cnt = 0;
	tstack->max_cnt = 0;
	tstack->tensors = malloc(sizeof(tensor_t*) * init_size);
	return tstack;
}

void tstack_free(struct tstack *tstack)
{
	free(tstack->tensors);
}


int tstack_cnt(struct tstack *tstack)
{
	return tstack->cnt;
}


void tstack_push(struct tstack *tstack, tensor_t *tensor)
{
	if (tstack->cnt == tstack->nelem) {
		tstack->nelem *= 2;
		tstack->tensors = realloc(tstack->tensors,
								sizeof(void*) * tstack->nelem);
	}
	tstack->tensors[tstack->cnt] = tensor;
	tstack->cnt++;

	if (tstack->max_cnt < tstack->cnt) {
		tstack->max_cnt = tstack->cnt;
	}
}

void tstack_pop(struct tstack *tstack)
{
	assert(tstack->cnt > 0);
	tstack->cnt--;
}

tensor_t* tstack_top(struct tstack *tstack)
{
	assert(tstack->cnt > 0);
	return tstack->tensors[(tstack->cnt)-1];
}

void* tstack_topmem(struct tstack *tstack)
{
	assert(tstack->cnt > 0);
	return tensor_mem(tstack->tensors[(tstack->cnt)-1]);
}


void tstack_clear(struct tstack *tstack)
{
	for (int i = 0; i < tstack->max_cnt; i++) {
		tensor_free(tstack->tensors[i]);
		tstack->tensors[i] = NULL;
	}
	tstack->cnt = 0;
	tstack->max_cnt = 0;
}

void tstack_clear_s(struct tstack *tstack)
{
	tstack->cnt = 0;
	tstack->max_cnt = 0;
}
