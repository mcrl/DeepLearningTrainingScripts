#ifndef __TSTACK_H__
#define __TSTACK_H__

#include "tensor.h"
struct tstack;

struct tstack* tstack_create(int init_size);
void tstack_free(struct tstack *tstack);
void tstack_push(struct tstack *tstack, tensor_t *tensor);
void tstack_pop(struct tstack *tstack);
tensor_t* tstack_top(struct tstack *tstack);
void* tstack_topmem(struct tstack *tstack);
int tstack_cnt(struct tstack *tstack);
void tstack_clear(struct tstack *tstack);
void tstack_clear_s(struct tstack *tstack);

#endif //__TSTACK_H__
