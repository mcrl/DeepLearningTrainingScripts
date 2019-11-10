#ifndef __RSTACK_H__
#define __RSTACK_H__
struct rstack;
struct rstack* rstack_create(int init_size);
void rstack_free(struct rstack *rstack);
void rstack_push(struct rstack *rstack, size_t size);
void rstack_pop(struct rstack *rstack);
size_t rstack_top_size(struct rstack *rstack);
void* rstack_top(struct rstack *rstack);
void rstack_clear(struct rstack *rstack);

#endif //__RSTACK_H__
