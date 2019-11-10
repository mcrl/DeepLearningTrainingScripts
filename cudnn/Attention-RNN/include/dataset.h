#ifndef __DATASET_H__
#define __DATASET_H__

#include "layers.h"

typedef struct _dataset_t dataset_t;

struct batch_t {
	int size;
	int max_src_len;
	int max_tgt_len;
	int *indices;
	int *src_len_array;
	int *tgt_len_array;
	tensor_t *input;
	tensor_t *target;
};

dataset_t* dataset_training(const char *input_path, const char *target_path, const char *input_dict, const char *target_dict);

void dataset_free_training(dataset_t *dataset);
void dataset_load_batch(dataset_t *dataset, struct batch_t *b, int iter);
void dataset_drop_batch(dataset_t *dataset, int iter);
int dataset_get_batch_size(dataset_t *dataset, int iter);
tensor_t* dataset_get_input(dataset_t *dataset, int iter);
tensor_t* dataset_get_target(dataset_t *dataset, int iter);

int dataset_nbatch(dataset_t *dataset);

// { dict
int dataset_input_len(dataset_t *dataset);
int dataset_target_len(dataset_t *dataset);
int dataset_input_padding_idx(dataset_t *dataset);
int dataset_target_padding_idx(dataset_t *dataset);
int dataset_dict_max_input_len(dataset_t *dataset);
int dataset_dict_max_target_len(dataset_t *dataset);
// }

#endif //__DATASET_H__
