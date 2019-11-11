#ifndef _PARAMS_H_
#define _PARAMS_H_

#include <stdbool.h>

#define FIXED_HEIGHT 161
#define NUM_LABEL 29
#define PRE_ALLOC_SIZE 509440240

typedef enum RNN_TYPE_e {
  GRU,
  RNN,
  LSTM
} RNN_TYPE;

typedef enum RNN_ACT_FUNC_TYPE_e {
  TANH,
  RELU
} RNN_ACT_FUNC_TYPE;

struct params {
  const long long seed;

  const long long rnn_hidden_size;
  const long long rnn_hidden_layers;
  const RNN_TYPE rnn_type;
  const RNN_ACT_FUNC_TYPE rnn_act_type;

  const long long epochs;
  const double learning_anneal;
  const double learning_rate;
  const double momentum;
  const double weight_decay;
  const long long max_norm;
  const long long l2; 
  const long long batch_size;
  const long long batch_size_eval;
  const bool augment;
  const bool exit_at_acc;
};

static const char char_table[NUM_LABEL] = {
  '_', '\'', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M',
  'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', ' '
};

static struct params params = { 
  .seed = 0xdeadbeef,

  .rnn_hidden_size = 800,
  .rnn_hidden_layers = 3,
  .rnn_type = GRU,
  .rnn_act_type = TANH,

  .epochs = 20,
  .learning_anneal = 1.01,
  .learning_rate = 0.0005,
  .momentum = 0.9,
  .weight_decay = 1e-5,
  .max_norm = 400,
  .l2 = 0,
  .batch_size = 20,
  .batch_size_eval = 8,
  .augment = true,
  .exit_at_acc = true
};

typedef struct wav_sample_s {
  int width;
  float *values;
} wav_sample;

typedef struct txt_sample_s {
  int width;
  int *values;
} txt_sample;

#endif
