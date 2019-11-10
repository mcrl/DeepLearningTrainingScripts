#ifndef __TENSOR_H__
#define __TENSOR_H__
#include <stddef.h>
#include <cudnn.h>


////////////////////////////////
// Light-weight tensor object //
////////////////////////////////

typedef struct _tensor_t tensor_t;

typedef struct {
	tensor_t *first;
	tensor_t *second;
} tensor_pair_t;

typedef struct {
	tensor_t *first;
	tensor_t *second;
	tensor_t *third;
} tensor_triple_t;

typedef struct {
	tensor_t *first;
	tensor_t *second;
	tensor_t *third;
	tensor_t *fourth;
} tensor_quadruple_t;

typedef enum {
	TENSOR_INT,
	TENSOR_FLOAT
} tensor_type_t;

typedef enum {
	TENSOR_CPU,
	TENSOR_GPU,
} tensor_dev_t;

tensor_t* tensor_create(const int sizes[], int dim);
tensor_t* tensor_create_empty(const int sizes[], int dim);
tensor_t* tensor_create_int(const int sizes[], int dim);
tensor_t* tensor_create_cpu(const int sizes[], int dim, tensor_type_t type);

tensor_t* tensor_samesize(tensor_t *tensor);

tensor_t* tensor_ones(const int sizes[], int dim);
tensor_t* tensor_zeros(const int sizes[], int dim);
tensor_t* tensor_uniform(const int sizes[], int dim, float min, float max);
tensor_t* tensor_rand(const int sizes[], int dim);

tensor_t* tensor_consti(const int sizes[], int dim, int val);
tensor_t* tensor_constf(const int sizes[], int dim, float val);


void tensor_free(tensor_t *tensor);
void tensor_free_s(tensor_t *tensor);
void tensor_allocate(tensor_t *tensor, int elem_size);
void tensor_init_consti(tensor_t *tensor, int val);
void tensor_init_constf(tensor_t *tensor, float val);

void* tensor_mem(tensor_t *tensor);
void* tensor_mem_idx(tensor_t *tensor, int idx);

tensor_dev_t tensor_dev(tensor_t *tensor);
tensor_type_t tensor_type(tensor_t *tensor);

cudnnTensorDescriptor_t tensor_descriptor(tensor_t *tensor);


float tensor_asum(tensor_t *t);
tensor_t* tensor_reduce_sum(tensor_t *t, int dim);
tensor_t* tensor_expand_sum(tensor_t *x, int dim, int size);
tensor_t *tensor_pointwise_mult(tensor_t *t, tensor_t *r);
tensor_t *tensor_pointwise_add(tensor_t *t, tensor_t *r);
tensor_pair_t tensor_split(tensor_t *x, const int* sizes, int dim);
tensor_t *tensor_pointwise_mult_sub(tensor_t *t, tensor_t *r);
void tensor_add(tensor_t *b, tensor_t *y);


void tensor_flatten(tensor_t *tensor);
void tensor_unsqueeze(tensor_t *tensor, int i);
void tensor_squeeze(tensor_t *tensor, int i);
void tensor_view(tensor_t *t, const int sizes[], int dim);


tensor_t* tensor_transpose(tensor_t *tensor, int i, int j);
void tensor_transpose_ip(tensor_t *tensor, int i, int j);
tensor_t* tensor_concat(tensor_t *x, tensor_t *y, int dim);
tensor_t *tensor_concat_all(tensor_t **ts, int size, int dim);
tensor_t *tensor_slice(tensor_t *x, int idx, int dim);




const int* tensor_sizes(tensor_t *t);
const int* tensor_strides(tensor_t *t);
int tensor_dim(tensor_t *t);
int tensor_nelem(tensor_t *t);
int tensor_size(tensor_t *tensor, int dim);
int tensor_stride(tensor_t *t, int dim);



void tensor_mem_set(tensor_t *t, void *m);


tensor_t *tensor_mask(tensor_t *x, int m);


void tensor_print_sizes(tensor_t *tensor);
void tensor_print(tensor_t *tensor);

void* get_workspace(size_t size);
void tensor_set_mem(tensor_t *x, void *mem);
void tensor_copy(tensor_t *src, tensor_t *dst);


void* tensor_cpu(tensor_t *x);

#endif //__TENSOR_H__
