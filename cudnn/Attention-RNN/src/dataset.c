/*
 * Dataset for Seq2Seq learning
 */

#include "dataset.h"
#include "gpu.h"
#include "snudnn.h"

#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>
#include <assert.h>
#include <string.h>


#define MAX(a, b) ((a) > (b)) ? (a) : (b);
struct _dataset_t {
	struct {
		int nbatches;
		struct batch_t *batches;
	} batch;

	struct {
		FILE *data_fp;
		int elem_size;
		struct {
			int size;
			int ssize;
			int *dims;
			uint64_t *data_offsets;
			uint64_t *dim_offsets;
			uint64_t *sizes;
		} input;

		struct {
			int size;
			int ssize;
			int *dims;
			uint64_t *data_offsets;
			uint64_t *dim_offsets;
			uint64_t *sizes;
		} target;
	} dataset;

	struct {
		int pad;
		int eos;
		int unk;
		int max_len;
		struct {
			int max_len;
			int size;
		} input;
		struct {
			int max_len;
			int size;
		} target;
	} dict;
};

#define MAX_SYMBOL_SIZE 128
static void dataset_init_dict(dataset_t *dataset, const char *input_dict, const char *target_dict)
{
	dataset->dict.pad = 1;
	dataset->dict.eos = 2;
	dataset->dict.unk = 3;

	char symbol[MAX_SYMBOL_SIZE];

	for (int i = 0; i < 2; i++) {
		FILE *fp = fopen(i == 0 ? input_dict : target_dict, "rb");

		int size = 4;
		int max_len = 0;
		int cnt;
		while (fscanf(fp, "%s %d", symbol, &cnt) != EOF) {
			int len = strlen(symbol);
			max_len = max_len > len ? max_len : len;
			size++;
		}

		if (i == 0) {
			dataset->dict.input.max_len = max_len;
			dataset->dict.input.size = size;
		} else {
			dataset->dict.target.max_len = max_len;
			dataset->dict.target.size = size;
		}
		fclose(fp);
	}
}

static void dataset_init_dataset(dataset_t *dataset, const char *input_path, const char *target_path)
{
	// fairseq indexed dataset:
	//
	// (1) 64bit magic number
	// (2) 64bit version
	// (3) 64bit type
	// 	   1: uint8
	// 	   2: int8
	// 	   3: int16
	// 	   4: int32
	// 	   5: int64
	// 	   6: float
	// 	   7: double
	// (4) 64bit element size
	// (5) 64bit size
	// (6) 64bit ssize
	// (7) dim offsets of size(5) + 1
	// (8) data offsets of size(5) + 1
	// (9) sizes of size(ssize)
	//
	//
	// data_offset[i+1] = data_offset[i] + product(sizes[d] for d in dim_offsets[i] ~ dim_offsets[i+1]
	// For sentence translation, dim_offsets[i+1] - dim_offsets[i] = 1

	static const int suffix_len = 4;
	static const char *suffix_idx = ".idx";
	static const char *suffix_bin = ".bin";


	for (int i = 0; i < 2; i++) {
		const char *path = i == 0 ? input_path : target_path;

		size_t path_len = strlen(path);
		char *file_name = (char *)malloc(sizeof(char) * (path_len + suffix_len + 1));

		strncpy(file_name, path, path_len);

		for (int i = 0; i <= suffix_len; i++) file_name[path_len + i] = suffix_idx[i];
		FILE *index_fp = fopen(file_name, "rb");
		assert(index_fp != NULL);

		for (int i = 0; i <= suffix_len; i++) file_name[path_len + i] = suffix_bin[i];
		dataset->dataset.data_fp = fopen(file_name, "rb");
		assert(dataset->dataset.data_fp != NULL);

		uint64_t magic;
		fread(&magic, sizeof(uint64_t), 1, index_fp);
		assert(magic == 0x584449544e54);

		uint64_t version;
		fread(&version, sizeof(uint64_t), 1, index_fp);
		assert(version == 0x1);

		uint64_t type, elem_size, size, ssize;
		fread(&type, sizeof(uint64_t), 1, index_fp);
		fread(&elem_size, sizeof(uint64_t), 1, index_fp);
		fread(&size, sizeof(uint64_t), 1, index_fp);
		fread(&ssize, sizeof(uint64_t), 1, index_fp);

		uint64_t *dim_offsets = (uint64_t *)malloc(sizeof(uint64_t) * (size + 1));
		uint64_t *data_offsets = (uint64_t *)malloc(sizeof(uint64_t) * (size + 1));
		uint64_t *sizes = (uint64_t *)malloc(sizeof(uint64_t) * ssize);

		size_t read = fread(dim_offsets,  sizeof(uint64_t), size + 1, index_fp);
		assert(read == size + 1);
		read = fread(data_offsets, sizeof(uint64_t), size + 1, index_fp);
		assert(read == size + 1);
		read = fread(sizes, sizeof(uint64_t), ssize, index_fp);
		assert(read == ssize);


		// FIXME: General case
		int *dims = malloc(sizeof(int) * size);
		assert(dim_offsets[0] == 0);
		for (int i = 0; i < size; i++) {
			assert(dim_offsets[i+1] - dim_offsets[i] == 1);
			dims[i] = sizes[i];
		}

		dataset->dataset.elem_size = elem_size;
		if (i == 0) {
			dataset->dataset.input.size = size;
			dataset->dataset.input.ssize = ssize;
			dataset->dataset.input.dims = dims;
			dataset->dataset.input.data_offsets = data_offsets;
			dataset->dataset.input.dim_offsets = dim_offsets;
			dataset->dataset.input.sizes = sizes;
		} else {
			dataset->dataset.target.size = size;
			dataset->dataset.target.ssize = ssize;
			dataset->dataset.target.dims = dims;
			dataset->dataset.target.data_offsets = data_offsets;
			dataset->dataset.target.dim_offsets = dim_offsets;
			dataset->dataset.target.sizes = sizes;
		}

		fclose(index_fp);
		free(file_name);
	}

}

static void shuffle(int *x, int len)
{
	for (int i = 0; i < len; i++) {
		//x[i] = rand() % (len - i); FIXME
		x[i] = i;
	}
}


static struct {
	size_t *src_sizes;
	size_t *tgt_sizes;
} c;

static int compar(const void *a, const void *b)
{

	int i = *(int*)a;
	int j = *(int*)b;

	if (c.src_sizes[i] == c.src_sizes[j]) {
		if (c.tgt_sizes[i] < c.tgt_sizes[j]) {
			return -1;
		} else if (c.tgt_sizes[i] == c.tgt_sizes[j]) {
			return 0;
		} else {
			return 1;
		}
	} else {
		if (c.src_sizes[i] < c.src_sizes[j]) {
			return -1;
		} else {
			return 1;
		}
	}

	/*
	   if (c.tgt_sizes[i] == c.tgt_sizes[j]) {
	   return c.src_sizes[i] - c.src_sizes[j];
	   } else {
	   return c.tgt_sizes[i] - c.tgt_sizes[j];
	   }
	   */
}

static void sort(dataset_t *dataset, int *indices, int indices_len)
{
	c.src_sizes = dataset->dataset.input.sizes;
	c.tgt_sizes = dataset->dataset.target.sizes;

	qsort(indices, indices_len, sizeof(int), compar);
}

static void dataset_prepare_batch(dataset_t *dataset, int max_toks)
{
	int indices_len = dataset->dataset.input.size; // FIXME
	int *indices = (int *)malloc(sizeof(int) * indices_len);
	int *offsets = (int *)malloc(sizeof(int) * (indices_len+1));

	size_t max_tokens = 0;
	int batch_len = 0;
	int offset = 0;
	size_t max_seq = 0;

	shuffle(indices, indices_len);
	sort(dataset, indices, indices_len);

	int nbatches = 0;
	offsets[0] = 0;
	for (int i = 0; i < indices_len; i++) {
		batch_len++;
		size_t si = dataset->dataset.input.sizes[indices[i]];
		size_t ti = dataset->dataset.target.sizes[indices[i]];

		if (max_seq < si) max_seq = si;
		if (max_seq < ti) max_seq = ti;

		size_t tokens = (si > ti ? si : ti);
		max_tokens = (max_tokens < tokens ? tokens : max_tokens); 

		if ((batch_len+1) * max_tokens > max_toks || i == indices_len - 1) {

			assert(batch_len > 0);

			// round off to 8
			batch_len = MAX(batch_len % 8, (batch_len / 8) * 8);
			offset += batch_len;
			offsets[1 + nbatches++] = offset;
			i = offset;
			max_tokens = 0;
			batch_len = 0;
		}
	}

	struct batch_t *batches = (struct batch_t *)malloc(sizeof(struct batch_t) * nbatches);
	int cur = rank;
	for (int i = 0; i < nbatches; i++) {
		batches[i].size = (offsets[i+1] - offsets[i]) / nrank;
		batches[i].indices = (int *)malloc(sizeof(int) * batches[i].size);
		batches[i].max_src_len = batches[i].max_tgt_len = 0;

		for (int j = 0; j < batches[i].size; j++) {
			batches[i].indices[j] = indices[cur];
			size_t si = dataset->dataset.input.sizes[batches[i].indices[j]];
			size_t ti = dataset->dataset.target.sizes[batches[i].indices[j]];

			batches[i].max_src_len = MAX(batches[i].max_src_len, si);
			batches[i].max_tgt_len = MAX(batches[i].max_tgt_len, ti);
			cur += nrank;
		}
	}

	dataset->batch.nbatches = nbatches;
	dataset->batch.batches = batches;

	free(offsets);
	free(indices);
}

static void dataset_init_training(dataset_t *dataset, const char *input_path, const char *target_path, const char *input_dict, const char *target_dict)
{
	int max_toks = 1200 * nrank;
	dataset_init_dict(dataset, input_dict, target_dict);
	dataset_init_dataset(dataset, input_path, target_path);
	dataset_prepare_batch(dataset, max_toks);
}

static void dataset_release(dataset_t *dataset) {
	free(dataset->dataset.input.dims);
	free(dataset->dataset.input.data_offsets);
	free(dataset->dataset.input.dim_offsets);
	free(dataset->dataset.input.sizes);

	free(dataset->dataset.target.dims);
	free(dataset->dataset.target.data_offsets);
	free(dataset->dataset.target.dim_offsets);
	free(dataset->dataset.target.sizes);
	fclose(dataset->dataset.data_fp);
}

dataset_t* dataset_training(const char *input_path, const char *target_path, const char *input_dict, const char *target_dict)
{
	dataset_t *dataset = malloc(sizeof(dataset_t));
	dataset_init_training(dataset, input_path, target_path, input_dict, target_dict);
	return dataset;
}

void dataset_free_training(dataset_t *dataset)
{
	dataset_release(dataset);
	free(dataset);
}

static void dataset_getitem(dataset_t *dataset, int i, void *input, void *target)
{
	assert(i >= 0);
	assert(i < dataset->dataset.input.size && i < dataset->dataset.target.size);

	size_t size = 1;

	for (int d = dataset->dataset.input.dim_offsets[i];
			d < dataset->dataset.input.dim_offsets[i + 1];
			d++) {
		size *= dataset->dataset.input.sizes[d];
	}
	assert(fseek(dataset->dataset.data_fp, dataset->dataset.input.data_offsets[i] * dataset->dataset.elem_size, SEEK_SET) == 0);
	fread(input, size * dataset->dataset.elem_size, 1, dataset->dataset.data_fp);
	for (int i = 0; i < size; i++) {
		((int *)input)[i] -= 1;
	}

	size = 1;

	for (int d = dataset->dataset.target.dim_offsets[i]; d < dataset->dataset.target.dim_offsets[i + 1]; d++) {
		size *= dataset->dataset.target.sizes[d];
	}

	assert(fseek(dataset->dataset.data_fp, dataset->dataset.target.data_offsets[i] * dataset->dataset.elem_size, SEEK_SET) == 0);
	fread(target, size * dataset->dataset.elem_size, 1, dataset->dataset.data_fp);

	// assume integer array
	for (int i = 0; i < size; i++) {
		((int *)target)[i] -= 1;
	}

}

void dataset_load_batch(dataset_t *dataset, struct batch_t *b, int iter)
{
	struct batch_t *batches = dataset->batch.batches;

	int max_src_len = batches[iter].max_src_len; 
	int max_tgt_len = batches[iter].max_tgt_len;
	int max_len = MAX(max_src_len, max_tgt_len);

	int *_input = (int *)malloc(sizeof(int) * max_len);
	int *_target = (int *)malloc(sizeof(int) * max_len);


	int isizes[] = { batches[iter].size, max_src_len };
	int tsizes[] = { batches[iter].size, max_tgt_len };
	tensor_t *input = tensor_create_int(isizes, 2);
	tensor_t *target = tensor_create_int(tsizes, 2);

	tensor_init_consti(input, dataset->dict.pad);
	tensor_init_consti(target, dataset->dict.pad);

	batches[iter].src_len_array = malloc(sizeof(int) * batches[iter].size);
	batches[iter].tgt_len_array = malloc(sizeof(int) * batches[iter].size);
	for (int i = 0; i < batches[iter].size; i++) {
		int idx = batches[iter].indices[i];
		size_t ss = dataset->dataset.input.sizes[idx];
		size_t ts = dataset->dataset.target.sizes[idx];
		batches[iter].src_len_array[i] = ss;
		batches[iter].tgt_len_array[i] = ts;

		dataset_getitem(dataset, idx,  _input, _target);

		chkCUDA(cudaMemcpyAsync(tensor_mem(input) + i * max_src_len * sizeof(int), _input,
					sizeof(int) * ss, cudaMemcpyHostToDevice, 0));
		chkCUDA(cudaMemcpyAsync(tensor_mem(target) + i * max_tgt_len * sizeof(int), _target,
					sizeof(int) * ts, cudaMemcpyHostToDevice, 0));
	}

	batches[iter].input = input;
	batches[iter].target = target;

	free(_input);
	free(_target);

	*b = batches[iter];
}

void dataset_drop_batch(dataset_t *dataset, int iter)
{
	struct batch_t *batches = dataset->batch.batches;

	tensor_free(batches[iter].input);
	tensor_free(batches[iter].target);

	free(batches[iter].src_len_array);
	free(batches[iter].tgt_len_array);
}

int dataset_nbatch(dataset_t *dataset)
{
	return dataset->batch.nbatches;
}

int dataset_input_len(dataset_t *dataset)
{
	return dataset->dict.input.size;
}

int dataset_target_len(dataset_t *dataset)
{
	return dataset->dict.target.size;
}

int dataset_input_padding_idx(dataset_t *dataset)
{
	return dataset->dict.pad;
}

int dataset_target_padding_idx(dataset_t *dataset)
{
	return dataset->dict.pad;
}

int dataset_dict_max_input_len(dataset_t *dataset)
{
	return dataset->dict.input.max_len;
}

int dataset_dict_max_target_len(dataset_t *dataset)
{
	return dataset->dict.target.max_len;
}
