/*
 * A GPU tensor Library
 */
#include "snudnn.h"
#include "tensor.h"
#include "gpu.h"
#include "cache_allocator.h"
#include "utils.h"

#include <cuda_runtime.h>
#include <cudnn.h>

#include <assert.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>


#include "kernels.h"


#define TENSOR_DIM_MAX 6
static void tensor_init_set(tensor_t *t, const int sizes[], int dim, tensor_type_t type, tensor_dev_t dev, void *mem);

void cuda_transpose_matrix(float *in, float *out, int i, int j, int nelem, int dim, int* strides);
void cuda_concat(const float *x, const float *y, float *z, int sizeX[], int sizeY[], int sizeZ[], int dim, int n);
void cuda_split(const float *z, float *x, float *y, int sizeZ[], int sizeX[], int sizeY[], int dim, int n) ;
void cuda_expand(const float *x, float *y, int sizeY[], int dim, int size, int n);
void cuda_slice(const float *x, float *y, const int sizes[], int N, int idx, int dim);
void cuda_mask(const int *x, int *y, int m, int N);

struct _tensor_t {
	size_t nelem;
	int dim;

	int sizes[TENSOR_DIM_MAX];
	int strides[TENSOR_DIM_MAX];

	int _cudnn_dim;
	int _cudnn_sizes[TENSOR_DIM_MAX];
	int _cudnn_strides[TENSOR_DIM_MAX];

	cudnnTensorDescriptor_t desc;

	bool initialized;
	bool allocated;

	tensor_type_t type;
	tensor_dev_t dev;

	void *mem;


	void *workspace; // cpu memory
	size_t workspace_size;

};


static cudnnDataType_t cudnn_type(tensor_type_t type)
{
	switch (type) {
		case TENSOR_INT:
			return CUDNN_DATA_INT32;
			break;
		case TENSOR_FLOAT:
			return CUDNN_DATA_FLOAT;
			break;
		default:
			break;
	}
	return CUDNN_DATA_FLOAT;
}

static void set_descriptor(tensor_t *t)
{
	cudnnDataType_t type = cudnn_type(t->type);

	int dim = t->dim < 3 ? 3 : t->dim;
	for (int i = t->dim; i < dim; i++)  {
		t->strides[i] = t->sizes[i] = 1;
	}

	chkCUDNN(cudnnSetTensorNdDescriptor(t->desc, type, dim, t->sizes, t->strides));

	cudnnDataType_t dataType;
	chkCUDNN(cudnnGetTensorNdDescriptor(t->desc, dim,
				&dataType, &t->_cudnn_dim,
				t->_cudnn_sizes, t->_cudnn_strides));
}

static void tensor_init_sizes(tensor_t *t, const int sizes[], int dim)
{
	size_t nelem = 1;
	for (int i = dim-1; i >= 0; i--) {
		t->sizes[i] = sizes[i];
		nelem *= sizes[i];
		t->strides[i] = (i == dim-1) ? 1 : t->strides[i+1] * sizes[i+1];
	}
	t->dim = dim;
	t->nelem = nelem;
}

static void tensor_init_set(tensor_t *t,
		const int sizes[],
		int dim,
		tensor_type_t type,
		tensor_dev_t dev,
		void *mem
		)
{
	// sizes: array of sizes of each dimension
	// dim: dimension

	memset(t, 0, sizeof(tensor_t));

	t->initialized = true;
	t->type = type;
	t->dev = dev;

	tensor_init_sizes(t, sizes, dim);

	chkCUDNN(cudnnCreateTensorDescriptor(&t->desc));
	set_descriptor(t);
	t->mem = mem;
	t->allocated = false;
}

static void tensor_init(tensor_t *t,
		const int sizes[],
		int dim,
		tensor_type_t type,
		tensor_dev_t dev)
{
	// sizes: array of sizes of each dimension
	// dim: dimension

	memset(t, 0, sizeof(tensor_t));

	t->initialized = true;
	t->allocated = true;
	t->type = type;
	t->dev = dev;

	tensor_init_sizes(t, sizes, dim);

	chkCUDNN(cudnnCreateTensorDescriptor(&t->desc));
	set_descriptor(t);
	if (t->dev == TENSOR_GPU) {
		chkCUDA(cacher_alloc(&(t->mem), sizeof(float)*t->nelem));
	} else {
		t->mem = malloc(sizeof(float)*t->nelem);
	}
	t->allocated = true;
}

static tensor_t* _tensor_create(const int sizes[], int dim, tensor_type_t type)
{
	// sizes: array of sizes of each dimension
	// dim: dimension
	// mem: pointer to memory to init

	tensor_t *tensor = malloc(sizeof(tensor_t));
	assert(tensor != NULL);
	tensor_init(tensor, sizes, dim, type, TENSOR_GPU);
	return tensor;
}

tensor_t* tensor_create_int(const int sizes[], int dim)
{
	return _tensor_create(sizes, dim, TENSOR_INT);
}

tensor_t* tensor_create(const int sizes[], int dim)
{
	return _tensor_create(sizes, dim, TENSOR_FLOAT);
}

tensor_t* tensor_create_empty(const int sizes[], int dim)
{
	tensor_t *tensor = malloc(sizeof(tensor_t));
	assert(tensor != NULL);
	tensor_init_set(tensor, sizes, dim, TENSOR_FLOAT, TENSOR_GPU, NULL);
	return tensor;
}

tensor_t* tensor_create_cpu(const int sizes[], int dim, tensor_type_t type)
{
	// sizes: array of sizes of each dimension
	// dim: dimension
	// mem: pointer to memory to init

	tensor_t *tensor = malloc(sizeof(tensor_t));
	assert(tensor != NULL);
	tensor_init(tensor, sizes, dim, type, TENSOR_CPU);
	return tensor;
}

tensor_t* tensor_ones(const int sizes[], int dim)
{
	tensor_t *tensor = _tensor_create(sizes, dim, TENSOR_FLOAT);
	tensor_init_constf(tensor, 1.0);
	return tensor;
}

tensor_t* tensor_zeros(const int sizes[], int dim)
{
	tensor_t *tensor = _tensor_create(sizes, dim, TENSOR_FLOAT);
	chkCUDA(cudaMemsetAsync(tensor->mem, 0, tensor->nelem * sizeof(float), 0));
	return tensor;
}

tensor_t* tensor_rand(const int sizes[], int dim)
{
	return tensor_uniform(sizes, dim, -1.0f, 1.0f);
}

tensor_t* tensor_uniform(const int sizes[], int dim, float min, float max)
{
	tensor_t *tensor = _tensor_create(sizes, dim, TENSOR_FLOAT);
	snudnn_uniform(tensor->mem, tensor->nelem, min, max);
	return tensor;
}

tensor_t* tensor_consti(const int sizes[], int dim, int val)
{
	tensor_t *t = tensor_create_int(sizes, dim);
	tensor_init_consti(t, val);

	return t;
}

tensor_t* tensor_constf(const int sizes[], int dim, float val)
{
	tensor_t *t = _tensor_create(sizes, dim, TENSOR_FLOAT);
	tensor_init_constf(t, val);
	return t;
}

tensor_t* tensor_samesize(tensor_t *tensor)
{
	if (!tensor)
		return NULL;
	tensor_t *t = _tensor_create(tensor->sizes, tensor->dim, tensor->type);
	return t;
}

void tensor_free_s(tensor_t *tensor)
{
	if (!tensor)
		return;

	chkCUDNN(cudnnDestroyTensorDescriptor(tensor->desc));

	free(tensor);
}


void tensor_free(tensor_t *tensor)
{
	if (!tensor)
		return;

	chkCUDNN(cudnnDestroyTensorDescriptor(tensor->desc));
	if (tensor->allocated)  {
		if (tensor->dev == TENSOR_GPU) {
			chkCUDA(cacher_free(tensor->mem));
		}
		else if (tensor->dev == TENSOR_CPU) {
			free(tensor->mem);
		}
	}

	free(tensor);
}


void tensor_init_consti(tensor_t *tensor, int val)
{
	assert(tensor->type == TENSOR_INT);
	if (tensor->dev == TENSOR_GPU) {
		snudnn_memseti(tensor->mem, tensor->nelem, val);
	} else {
		int *tmp = tensor->mem;
		for (int i = 0; i < tensor->nelem; i++) {
			tmp[i] = val;
		}
	}
}

void tensor_init_constf(tensor_t *tensor, float val)
{
	assert(tensor->type == TENSOR_FLOAT);
	if (tensor->dev == TENSOR_GPU) {
		snudnn_memsetf(tensor->mem, tensor->nelem, val);
	} else {
		float *tmp = tensor->mem;
		for (int i = 0; i < tensor->nelem; i++) {
			tmp[i] = val;
		}
	}
}

int tensor_size(tensor_t *t, int dim)
{
	assert(dim < t->dim);
	return t->sizes[dim];
}

int tensor_stride(tensor_t *t, int dim)
{
	assert(dim < t->dim);
	return t->strides[dim];
}

void* tensor_mem(tensor_t *tensor)
{
	if (tensor == NULL)
		return NULL;
	return tensor->mem;
}

void* tensor_mem_idx(tensor_t *tensor, int idx)
{
	assert(tensor->allocated);
	int offset = 0;

	switch (tensor->type) {
		case TENSOR_FLOAT:
			offset = idx * sizeof(float); break;
		case TENSOR_INT:
			offset = idx * sizeof(int); break;
		default: break;
	}
	return tensor->mem + offset;
}

tensor_dev_t tensor_dev(tensor_t *tensor)
{
	return tensor->dev;
}

tensor_type_t tensor_type(tensor_t *tensor)
{
	return tensor->type;
}


static void tensor_print_descriptor(cudnnTensorDescriptor_t desc)
{
	int dimA[TENSOR_DIM_MAX];
	int strideA[TENSOR_DIM_MAX];

	cudnnDataType_t dataType;
	int nbDims;
	chkCUDNN(cudnnGetTensorNdDescriptor(desc, TENSOR_DIM_MAX, &dataType, &nbDims, dimA, strideA));
	printf("(");
	for (int i = 0; i < nbDims; i++) {
		printf("%d", dimA[i]);
		if (i < nbDims-1)
			printf(", ");
	}
	printf(")");
}

void tensor_print_sizes(tensor_t *tensor)
{
	if (!tensor) {
		printf("(null)\n");
		return;
	}
	tensor_print_descriptor(tensor->desc);
	printf(" // (");
	for (int i = 0; i < tensor->dim; i++) {
		printf("%d", tensor->strides[i]);
		if (i < tensor->dim-1)
			printf(", ");
	}
	printf(")");

	printf(" // (");
	for (int i = 0; i < tensor->dim; i++) {
		printf("%d", tensor->sizes[i]);
		if (i < tensor->dim-1)
			printf(", ");
	}
	printf(")\n");
}

void tensor_print(tensor_t *tensor)
{
	tensor_print_sizes(tensor);

	void *mem = tensor->mem;
	if (tensor->dev == TENSOR_GPU) {
		mem = malloc(sizeof(float) * tensor->nelem);
		chkCUDA(cudaMemcpy(mem, tensor->mem, sizeof(float) * tensor->nelem, cudaMemcpyDeviceToHost));
	}

	for (int i = 0; i < tensor->nelem; i++) {
		if (tensor->type == TENSOR_INT) {
			printf("%d", ((int*)(mem))[i]);
		} else {
			printf("%f", ((float*)(mem))[i]);
		}
		if (i < tensor->nelem - 1)
			printf(", ");
	}
	printf("\n");
	if (tensor->dev == TENSOR_GPU) {
		free(mem);
	}
}

cudnnTensorDescriptor_t tensor_descriptor(tensor_t *tensor)
{
	if (!tensor)
		return NULL;
	assert(tensor->dev == TENSOR_GPU);
	return tensor->desc;
}

void tensor_transpose_ip(tensor_t *tensor, int i, int j)
{
	tensor_t *out = tensor_transpose(tensor, i, j);
	*tensor = *out;
	out->allocated = false;
	tensor_free(out);
}

tensor_t* tensor_transpose(tensor_t *tensor, int i, int j)
{
	// in-place operation
	assert(i < tensor->dim && j < tensor->dim);

	int sizes[TENSOR_DIM_MAX];
	memcpy(sizes, tensor->sizes, sizeof(int)*TENSOR_DIM_MAX);
	sizes[i] = tensor->sizes[j];
	sizes[j] = tensor->sizes[i];
	tensor_t *y = _tensor_create(sizes, tensor->dim, tensor->type);

	cuda_transpose_matrix(tensor->mem, y->mem, i, j, tensor->nelem, y->dim, tensor->strides);

	return y;
}

void tensor_flatten(tensor_t *t)
{
	int sizes[] = { t->nelem };
	tensor_init_sizes(t, sizes, 1);
	set_descriptor(t);
}

void tensor_unsqueeze(tensor_t *tensor, int j)
{
	assert(j >= 0);
	for (int i = tensor->dim; i > j; i--) {
		tensor->sizes[i] = tensor->sizes[i-1];
	}
	tensor->sizes[j] = 1;
	tensor->dim++;
	tensor->strides[tensor->dim-1] = 1;
	for (int i = tensor->dim-2; i >= 0; i--) {
		tensor->strides[i] = tensor->strides[i+1] * tensor->sizes[i+1];
	}

	set_descriptor(tensor);
}

void tensor_squeeze(tensor_t *tensor, int j)
{
	assert(j >= 0);
	assert(tensor->sizes[j] == 1);
	assert(tensor->strides[tensor->dim-1] == 1);

	for (int i = j; i < tensor->dim; i++) {
		tensor->sizes[i] = tensor->sizes[i+1];
	}
	tensor->dim--;
	tensor->strides[tensor->dim-1] = 1;
	for (int i = tensor->dim-2; i >= 0; i--) {
		tensor->strides[i] = tensor->strides[i+1] * tensor->sizes[i+1];
	}
	set_descriptor(tensor);
}


void tensor_add(tensor_t *b, tensor_t *y)
{
	assert(y->dim >= b->dim);

	int dim = y->dim;
	int bdim = b->dim;
	int ytmp[dim];
	int btmp[bdim];
	for (int i = 0; i < dim; i++) {
		ytmp[i] = y->sizes[i];
	}
	for (int i = 0; i < bdim; i++) {
		btmp[i] = b->sizes[i];
	}

	int n = y->sizes[dim-1];
	assert(n == btmp[bdim-1]);

	int sizes[] = { tensor_nelem(y) / n, 1, 1, n };
	int bsizes[] = { 1, 1, 1, n };
	tensor_view(y, sizes, 4);
	tensor_view(b, bsizes, 4);

	chkCUDNN(cudnnAddTensor(cudnn_handle,
				&_one, tensor_descriptor(b), tensor_mem(b),
				&_one, tensor_descriptor(y), tensor_mem(y)
				));
	tensor_view(y, ytmp, dim);
	tensor_view(b, btmp, bdim);
}

void tensor_view(tensor_t *t, const int sizes[], int dim)
{
	int nelem = 1;
	for (int i = 0; i < dim; i++) {
		nelem *= sizes[i];
	}
	assert (nelem == t->nelem);
	tensor_init_sizes(t, sizes, dim);
	set_descriptor(t);
}

tensor_t* tensor_concat(tensor_t *x, tensor_t *y, int dim)
{
	assert(x->dim == y->dim);
	assert(x->type == y->type);

	int zsizes[x->dim];
	for (int i = 0; i < x->dim; i++) {
		assert((i == dim) || x->sizes[i] == y->sizes[i]);
		zsizes[i] = x->sizes[i];
	}
	zsizes[dim] += y->sizes[dim];

	tensor_t *z = _tensor_create(zsizes, x->dim, x->type);
	cuda_concat(x->mem, y->mem, z->mem, x->sizes, y->sizes, z->sizes, dim, z->dim);

	return z;
}


tensor_pair_t tensor_split(tensor_t *z, const int *xsizes, int dim)
{
	assert(dim < z->dim);
	int ysizes[z->dim];
	for (int i = 0; i < z->dim; i++) {
		ysizes[i] = z->sizes[i];
		if (i == dim)
			ysizes[i] -= xsizes[i];
	}

	tensor_t *x = _tensor_create(xsizes, z->dim, z->type);
	tensor_t *y = _tensor_create(ysizes, z->dim, z->type);

	cuda_split(z->mem, x->mem, y->mem, z->sizes, x->sizes, y->sizes, dim, z->dim);

	tensor_pair_t pair = { .first = x, .second = y };
	return pair;
}

int tensor_dim(tensor_t *t)
{
	return t->dim;
}

const int* tensor_sizes(tensor_t *t)
{
	return t->sizes;
}

const int* tensor_strides(tensor_t *t)
{
	return t->strides;
}


int tensor_nelem(tensor_t *t)
{
	if (!t)
		return 0;
	return t->nelem;
}

float tensor_asum(tensor_t *t)
{
	float result;
	chkCUDA(cudaDeviceSynchronize());
	chkCUBLAS(cublasSasum(cublas_handle, tensor_nelem(t), tensor_mem(t), 1, &result));
	chkCUDA(cudaDeviceSynchronize());
	return result;
}

void tensor_mem_set(tensor_t *t, void *m)
{
	t->mem = m;
}

void* get_workspace(size_t size)
{
	/*
	void *workspace;
	cacher_alloc(&workspace, size);
	return workspace;
	*/
	static size_t _size = 1024*1024*512;
	static void *workspace = NULL;
	if (workspace == NULL) {
		chkCUDA(cudaMalloc(&workspace, _size));
		//chkCUDA(cacher_alloc(&workspace, _size));
	}

	if (size > _size) {
		if (_size > 0) {
			chkCUDA(cudaFree(workspace));
		}
		_size = size;
		//chkCUDA(cacher_alloc(&workspace, size));
		chkCUDA(cudaMalloc(&workspace, size));
	}

	return workspace;
}

tensor_t* tensor_reduce_sum(tensor_t *t, int dim)
{
	static cudnnReduceTensorDescriptor_t reduce_desc = NULL;

	if (reduce_desc == NULL) {
		chkCUDNN(cudnnCreateReduceTensorDescriptor(&reduce_desc));
		chkCUDNN(cudnnSetReduceTensorDescriptor(reduce_desc,
					CUDNN_REDUCE_TENSOR_ADD, CUDNN_DATA_FLOAT,
					CUDNN_NOT_PROPAGATE_NAN, CUDNN_REDUCE_TENSOR_NO_INDICES,
					CUDNN_32BIT_INDICES));
	}
	int sizes[t->dim];
	for (int i = 0; i < t->dim; i++) {
		sizes[i] = t->sizes[i];
	}
	sizes[dim] = 1;



	tensor_t *n = _tensor_create(sizes, t->dim, t->type);

	size_t size;
	chkCUDNN(cudnnGetReductionWorkspaceSize(cudnn_handle,
				reduce_desc, tensor_descriptor(t), tensor_descriptor(n), &size));
	void *workspace = get_workspace(size);
	chkCUDNN(cudnnReduceTensor(cudnn_handle,
				reduce_desc, NULL, 0, workspace, size,
				&_one, tensor_descriptor(t), tensor_mem(t),
				&_zero, tensor_descriptor(n), tensor_mem(n)));

	tensor_squeeze(n, dim);
	return n;
}

tensor_t* tensor_expand_sum(tensor_t *x, int dim, int size)
{
	assert(dim >= 0);

	int sizes[x->dim];
	for (int i = 0; i < dim; i++) {
		sizes[i] = x->sizes[i];
	}
	sizes[dim] = size;
	for (int i = dim+1; i < x->dim+1; i++) {
		sizes[i] = x->sizes[i-1];
	}

	tensor_t *y = _tensor_create(sizes, x->dim+1, x->type);

	cuda_expand(tensor_mem(x), tensor_mem(y), y->sizes, dim, size, y->dim);

	return y;
}

static bool is_samesize(tensor_t *x, tensor_t *y)
{
	if (x->dim != y->dim)
		return false;

	for (int i = 0; i < x->dim; i++) {
		if (x->sizes[i] != y->sizes[i])
			return false;
	}
	return true;
}


static tensor_t* apply_pointwise(tensor_t *x, tensor_t *y, cudnnOpTensorDescriptor_t desc)
{
	tensor_t *z = NULL;
	bool found = false;
	int squeeze = 0;
	assert(x->dim == y->dim);
	if (x->dim <= 3) {
		squeeze = 4 - x->dim;
	}

	for (int i = 0; i < squeeze; i++) {
		tensor_unsqueeze(x, 0);
		tensor_unsqueeze(y, 0);
	}

	for (int i = 0; i < x->_cudnn_dim; i++) {
		if (x->_cudnn_sizes[i] != y->_cudnn_sizes[i]) {
			if (found)
				assert(false);

			found = true;
			if (x->_cudnn_sizes[i] > y->_cudnn_sizes[i]) {
				z = _tensor_create(x->sizes, x->dim, x->type);
			} else {
				z = _tensor_create(y->sizes, y->dim, y->type);
			}
		}
	}
	if (!found)
		z = _tensor_create(x->sizes, x->dim, x->type);


	chkCUDNN(cudnnOpTensor(cudnn_handle, desc,
				&_one, tensor_descriptor(x), tensor_mem(x),
				&_one, tensor_descriptor(y), tensor_mem(y),
				&_zero, tensor_descriptor(z), tensor_mem(z)));

	for (int i = 0; i < squeeze; i++) {
		tensor_squeeze(x, 0);
		tensor_squeeze(y, 0);
		tensor_squeeze(z, 0);
	}

	return z;
}

tensor_t *tensor_pointwise_mult(tensor_t *x, tensor_t *y)
{
	assert(x->dim == y->dim);
	static cudnnOpTensorDescriptor_t op_desc = NULL;

	if (op_desc == NULL) {
		chkCUDNN(cudnnCreateOpTensorDescriptor(&op_desc));
		chkCUDNN(cudnnSetOpTensorDescriptor(op_desc,
					CUDNN_OP_TENSOR_MUL, CUDNN_DATA_FLOAT, CUDNN_NOT_PROPAGATE_NAN));
	}
	return apply_pointwise(x, y, op_desc);
}

tensor_t *tensor_pointwise_mult_sub(tensor_t *x, tensor_t *y)
{
	assert(x->dim == y->dim);
	static cudnnOpTensorDescriptor_t op_desc = NULL;

	if (op_desc == NULL) {
		chkCUDNN(cudnnCreateOpTensorDescriptor(&op_desc));
		chkCUDNN(cudnnSetOpTensorDescriptor(op_desc,
					CUDNN_OP_TENSOR_MUL, CUDNN_DATA_FLOAT, CUDNN_NOT_PROPAGATE_NAN));
	}

	return apply_pointwise(x, y, op_desc);
}

tensor_t *tensor_pointwise_add(tensor_t *x, tensor_t *y)
{
	static cudnnOpTensorDescriptor_t op_desc = NULL;
	if (op_desc == NULL) {
		chkCUDNN(cudnnCreateOpTensorDescriptor(&op_desc));
		chkCUDNN(cudnnSetOpTensorDescriptor(op_desc,
					CUDNN_OP_TENSOR_ADD, CUDNN_DATA_FLOAT, CUDNN_NOT_PROPAGATE_NAN));
	}
	return apply_pointwise(x, y, op_desc);
}


tensor_t *tensor_mask(tensor_t *x, int m)
{
	tensor_t *y = _tensor_create(x->sizes, x->dim, x->type);
	cuda_mask(tensor_mem(x), tensor_mem(y), m, tensor_nelem(y));
	return y;
}


tensor_t *tensor_concat_all(tensor_t **ts, int size, int dim)
{
	assert(size > 1);
	tensor_t *x = tensor_concat(ts[0], ts[1], dim);
	for (int i = 2; i < size; i++) {
		tensor_t *t = tensor_concat(x, ts[i], dim);
		tensor_free(x);
		x = t;
	}
	return x;
}

static tensor_t* tensor_slice0(tensor_t *x, int idx)
{
	tensor_t *tensor = malloc(sizeof(tensor_t));
	assert(tensor != NULL);
	assert(x->type == TENSOR_FLOAT);
	int sizes[TENSOR_DIM_MAX];
	for (int i = 1; i < x->dim; i++) {
		sizes[i] = x->sizes[i];
	}
	sizes[0] = 1;
	tensor_init_set(tensor, sizes, x->dim, x->type, TENSOR_GPU, tensor_mem(x) + sizeof(float) * tensor_stride(x, 0));
	tensor_squeeze(tensor, 0);
	return tensor;
}

tensor_t *tensor_slice(tensor_t *x, int idx, int dim)
{
	assert(dim < x->dim);
	assert(idx < x->sizes[dim]);

	if (dim == 0)
		return tensor_slice0(x, idx);

	int sizes[TENSOR_DIM_MAX];
	for (int i = 0; i < x->dim; i++) {
		sizes[i] = x->sizes[i];
	}
	sizes[dim] = 1;

	tensor_t *y = _tensor_create(sizes, x->dim, x->type);
	cuda_slice(tensor_mem(x), tensor_mem(y), x->sizes, x->dim, idx, dim);
	tensor_squeeze(y, dim);
	return y;
}


void tensor_set_mem(tensor_t *x, void *mem)
{
	if (x->allocated)
		chkCUDA(cacher_free(x->mem));
	x->allocated = true;
	x->mem = mem;
}


void tensor_copy(tensor_t *src, tensor_t *dst)
{
	chkCUDA(cudaMemcpyAsync(tensor_mem(dst), tensor_mem(src),
				tensor_nelem(dst) * sizeof(float),
				cudaMemcpyDeviceToDevice, 0));
}

void* tensor_cpu(tensor_t *x)
{
	void *buf = malloc(sizeof(float) * tensor_nelem(x));
	chkCUDA(cudaMemcpy(buf, tensor_mem(x),
				tensor_nelem(x) * sizeof(float),
				cudaMemcpyDeviceToHost));
	return buf;
}
