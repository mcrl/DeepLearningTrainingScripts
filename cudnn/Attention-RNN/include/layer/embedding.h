#ifndef _EMBEDDING_H_
#define _EMBEDDING_H_

#include "layer.h"
#include "dict.h"

/*
 * Implementation of torch.nn.Embedding
 */

typedef struct embedding_param {
  int num_embeddings;
  int embedding_dim;
  int padding_idx;
  float embedding_scale;
} embedding_param_t;

typedef struct embedding {
  LAYER_INHERIT;

  int num_embeddings;
  int embedding_dim;
  int padding_idx;
  float embedding_scale;

  size_t weight_size;

  float *weight;
  float *d_weight;

  size_t off_weight;

  /* unused 
  float max_norm;
  float norm_type;
  int scale_grad_by_freq;
  int sparse;
  */

  size_t batch_size;
  size_t seq_len;

} embedding_t;

LAYER_CREATE(embedding);
LAYER_DESTROY(embedding);

LAYER_INIT(embedding);
LAYER_INIT_WEIGHTS(embedding);
LAYER_SET(embedding);
LAYER_FORWARD(embedding);
LAYER_BACKWARD(embedding);
LAYER_FREE(embedding);

#endif /* _EMBEDDING_H_ */

