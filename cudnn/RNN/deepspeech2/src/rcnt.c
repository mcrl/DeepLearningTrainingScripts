#include <stdbool.h>
#include <math.h>
#include <time.h>

#include <builtin_types.h>
#include <cuda_runtime_api.h>
#include <cuda.h>
#include <cudnn.h>

#include "deepspeech.h"
#include "deepspeech_cuda.h"
#include "utils.h"
#include "params.h"

#include "rcnt.h"

void init_rcnt_layer(rcnt_layer *rcl, cudnnHandle_t cudnn, cudnnRNNMode_t mode,
  int input_size, int hidden_size, int max_batch_size, int max_seq,
  bool bn_first, float *weights, float *biases, float *bn_scale, float *bn_bias,
  int *off_d)
{
  cudnnDropoutDescriptor_t dropoutDesc;
  size_t state_size;
  void *states;
  size_t weight_bytes;

  rcl->cudnn = cudnn;
  rcl->zero = 0.0f;
  rcl->mode = mode;
  rcl->input_size = input_size;
  rcl->hidden_size = hidden_size;
  rcl->cnt_seqs = (int *)malloc(sizeof(int) * max_seq);
  rcl->bn_first = bn_first;
  rcl->_input_rcnt_desc = (cudnnTensorDescriptor_t *)
      malloc(sizeof(cudnnTensorDescriptor_t) * max_seq);

  rcl->weight = rcl->bn_scale = rcl->bn_bias = NULL;
  rcl->d_weight = rcl->d_bn_scale = rcl->d_bn_bias = NULL;

  if (rcl->bn_first == true) {
    chkCUDNN(cudnnCreateTensorDescriptor(&rcl->bn_desc));
    chkCUDNN(cudnnCreateTensorDescriptor(&rcl->before_bn_desc)); 
    chkCUDNN(cudnnCreateTensorDescriptor(&rcl->after_bn_desc));
    
    MALLOC_TENSOR_FLOATZ(&rcl->bn_result_running_mean, 1, 1, 1, rcl->input_size);
    MALLOC_TENSOR_FLOATZ(&rcl->bn_result_running_var, 1, 1, 1, rcl->input_size);
    float *tmp = malloc(sizeof(float) * rcl->input_size);
    for (int i = 0; i < rcl->input_size; i++) {
      tmp[i] = 1.0f;
    }
    chkCUDA(cudaMemcpy(rcl->bn_result_running_var, tmp, sizeof(float) * rcl->input_size, cudaMemcpyHostToDevice));
    free(tmp);

    chkCUDNN(cudnnCreateTensorDescriptor(&rcl->d_bn_desc));
    chkCUDNN(cudnnCreateTensorDescriptor(&rcl->d_before_bn_desc)); 
    chkCUDNN(cudnnCreateTensorDescriptor(&rcl->d_after_bn_desc));

    rcl->off_d_bn_scale = *off_d;
    *off_d += rcl->input_size;
    rcl->off_d_bn_bias = *off_d;
    *off_d += rcl->input_size;
  }

  /* set dummy */
  int dim_input[3] = {max_batch_size, rcl->input_size, 1};
  int stride_input[3] = {rcl->input_size, 1, 1};
  chkCUDNN(cudnnCreateTensorDescriptor(&rcl->x));
  chkCUDNN(cudnnSetTensorNdDescriptor(rcl->x, CUDNN_DATA_FLOAT, 3,
    dim_input, stride_input));

  /* create input & output tensor desc */
  chkCUDNN(cudnnCreateRNNDataDescriptor(&rcl->input_rcnt_desc));
  chkCUDNN(cudnnCreateRNNDataDescriptor(&rcl->after_rcnt_desc));
  chkCUDNN(cudnnCreateRNNDataDescriptor(&rcl->d_input_rcnt_desc));
  chkCUDNN(cudnnCreateRNNDataDescriptor(&rcl->d_after_rcnt_desc));
  for (int i = 0; i < max_seq; i++) {
    chkCUDNN(cudnnCreateTensorDescriptor(&rcl->_input_rcnt_desc[i]));
  }

  /* create rcnt layer's desc */
  chkCUDNN(cudnnCreateRNNDescriptor(&rcl->rcnt_desc));
  chkCUDNN(cudnnCreateTensorDescriptor(&rcl->hs_output_desc));
  chkCUDNN(cudnnCreateFilterDescriptor(&rcl->weight_desc));

  chkCUDNN(cudnnCreateTensorDescriptor(&rcl->d_hs_output_desc));
  chkCUDNN(cudnnCreateFilterDescriptor(&rcl->d_weight_desc));

  chkCUDNN(cudnnCreateReduceTensorDescriptor(&rcl->reduce_desc));
  chkCUDNN(cudnnCreateTensorDescriptor(&rcl->a_desc));
  chkCUDNN(cudnnCreateTensorDescriptor(&rcl->c_desc));

  /* set dropout desc */
  chkCUDNN(cudnnCreateDropoutDescriptor(&dropoutDesc));
  chkCUDNN(cudnnDropoutGetStatesSize(rcl->cudnn, &state_size));
  chkCUDA(cudaMalloc(&states, state_size));
  chkCUDNN(cudnnSetDropoutDescriptor(dropoutDesc,
    rcl->cudnn, 0, states, state_size, 1237ull));

  /* set rnn desc */
  chkCUDNN(cudnnSetRNNDescriptor_v6(rcl->cudnn, rcl->rcnt_desc,
    rcl->hidden_size, 1, dropoutDesc, CUDNN_LINEAR_INPUT,
    CUDNN_BIDIRECTIONAL, mode, CUDNN_RNN_ALGO_STANDARD, CUDNN_DATA_FLOAT));
  chkCUDNN(cudnnSetRNNPaddingMode(rcl->rcnt_desc, CUDNN_RNN_PADDED_IO_ENABLED));

  /* get param size, set and alloc weight */
  chkCUDNN(cudnnGetRNNParamsSize(rcl->cudnn, rcl->rcnt_desc,
    rcl->x, &weight_bytes, CUDNN_DATA_FLOAT)); 
  rcl->dim_weight[0] = weight_bytes / sizeof(float);
  rcl->dim_weight[1] = rcl->dim_weight[2] = 1;

  chkCUDNN(cudnnSetFilterNdDescriptor(rcl->weight_desc,
    CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, 3, rcl->dim_weight));

  chkCUDNN(cudnnSetFilterNdDescriptor(rcl->d_weight_desc,
    CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, 3, rcl->dim_weight));

  rcl->off_d_weight = *off_d;
  *off_d += rcl->dim_weight[0];

  if (weights != NULL && biases != NULL) {
    set_rcnt_layer_params(rcl, weights, biases);
  }
  if (bn_scale != NULL && bn_bias != NULL) {
    chkCUDA(cudaMemcpy(rcl->bn_scale, bn_scale,
      sizeof(float) * rcl->input_size, cudaMemcpyHostToDevice));
    chkCUDA(cudaMemcpy(rcl->bn_bias, bn_bias,
      sizeof(float) * rcl->input_size, cudaMemcpyHostToDevice));
  }

  rcl->buf_d_bn_scale = NULL;
  rcl->buf_d_bn_bias = NULL;
  rcl->buf_d_weight = NULL;

  MALLOC_TENSOR_FLOATZ(&rcl->packed_seq, max_batch_size,
    rcl->input_size, max_seq, 1);
  MALLOC_TENSOR_FLOATZ(&rcl->after_rcnt, max_batch_size,
    rcl->hidden_size * 2, max_seq, 1);
  MALLOC_TENSOR_FLOATZ(&rcl->summed_seq, max_batch_size,
    rcl->hidden_size, max_seq, 1);
  rcl->output = rcl->summed_seq;

  MALLOC_TENSOR_FLOATZ(&rcl->d_input, max_batch_size,
    rcl->input_size, max_seq, 1);
  MALLOC_TENSOR_FLOATZ(&rcl->d_after_rcnt, max_batch_size,
    rcl->hidden_size * 2, max_seq, 1);
  MALLOC_TENSOR_FLOATZ(&rcl->d_padded_seq, max_batch_size,
    rcl->hidden_size * 2, max_seq, 1);

  MALLOC_TENSOR_FLOAT(&rcl->hs_output, max_batch_size, 2,
    rcl->hidden_size, 1);
  MALLOC_TENSOR_FLOAT(&rcl->d_hs_output, max_batch_size, 2,
    rcl->hidden_size, 1);

  if (rcl->bn_first == true) {
    MALLOC_TENSOR_FLOATZ(&rcl->input, max_batch_size,
      rcl->input_size, max_seq, 1);
    MALLOC_TENSOR_FLOATZ(&rcl->d_before_bn, max_batch_size * max_seq,
      1, 1, rcl->input_size);
  }
}

void set_rcnt_layer(rcnt_layer *rcl, float *input, float **output,
  float *d_output, float **d_input, int batch_size, int seq_length,
  int *cnt_seqs, int *seqs, bool is_training)
{
  chkCUDA(cudaMemset(rcl->after_rcnt, 0,
    sizeof(float) * batch_size * seq_length * rcl->hidden_size * 2));
  chkCUDA(cudaMemset(rcl->d_padded_seq, 0,
    sizeof(float) * batch_size * seq_length * rcl->hidden_size * 2));
  chkCUDA(cudaMemset(rcl->d_input, 0,
    sizeof(float) * batch_size * seq_length * rcl->input_size));

  if (rcl->weight == NULL) {
    rcl->d_weight = d + rcl->off_d_weight;
    rcl->weight = p + rcl->off_d_weight;

    float k = sqrt(1.0 / rcl->hidden_size);
    INITIALIZE_TENSOR_URAND(rcl->weight, -k, k, rcl->dim_weight[0]);

    if (rcl->bn_first == true) {
      rcl->d_bn_scale = d + rcl->off_d_bn_scale;
      rcl->d_bn_bias = d + rcl->off_d_bn_bias;

      rcl->bn_scale = p + rcl->off_d_bn_scale;
      rcl->bn_bias = p + rcl->off_d_bn_bias;

      INITIALIZE_TENSOR_URAND(rcl->bn_scale, 0, 1, rcl->input_size);
    }
  }

  size_t weight_bytes;

  rcl->is_training = is_training;

  rcl->batch_size = batch_size;
  rcl->input = input;
  rcl->seq_length = seq_length;
  rcl->d_summed_seq = d_output;

  rcl->dim_hidden_state[0] = 2;
  rcl->dim_hidden_state[1] = batch_size;
  rcl->dim_hidden_state[2] = rcl->hidden_size;
  rcl->stride_hidden_state[0] =
    rcl->dim_hidden_state[2] * rcl->dim_hidden_state[1];
  rcl->stride_hidden_state[1] = rcl->dim_hidden_state[2];
  rcl->stride_hidden_state[2] = 1;

  rcl->dim_input[1] = rcl->input_size;
  rcl->dim_input[2] = 1;
  rcl->dim_output[1] = rcl->hidden_size * 2;
  rcl->dim_output[2] = 1;
  rcl->stride_input[0] = rcl->dim_input[2] * rcl->dim_input[1];
  rcl->stride_input[1] = rcl->dim_input[2];
  rcl->stride_input[2] = 1;
  rcl->stride_output[0] = rcl->dim_output[2] * rcl->dim_output[1];
  rcl->stride_output[1] = rcl->dim_output[2];
  rcl->stride_output[2] = 1;

  if (rcl->bn_first == true) {
    rcl->before_bn = input;
    chkCUDNN(cudnnSetTensor4dDescriptor(rcl->before_bn_desc, CUDNN_TENSOR_NCHW,
      CUDNN_DATA_FLOAT, batch_size * seq_length, rcl->input_size, 1, 1));
    chkCUDNN(cudnnSetTensor4dDescriptor(rcl->after_bn_desc, CUDNN_TENSOR_NCHW,
      CUDNN_DATA_FLOAT, batch_size * seq_length, rcl->input_size, 1, 1));
    chkCUDNN(cudnnDeriveBNTensorDescriptor(rcl->bn_desc, rcl->before_bn_desc,
      CUDNN_BATCHNORM_PER_ACTIVATION));
    
    chkCUDNN(cudnnSetTensor4dDescriptor(rcl->d_before_bn_desc, CUDNN_TENSOR_NCHW,
      CUDNN_DATA_FLOAT, batch_size * seq_length, rcl->input_size, 1, 1));
    chkCUDNN(cudnnSetTensor4dDescriptor(rcl->d_after_bn_desc, CUDNN_TENSOR_NCHW,
      CUDNN_DATA_FLOAT, batch_size * seq_length, rcl->input_size, 1, 1));
    chkCUDNN(cudnnDeriveBNTensorDescriptor(rcl->d_bn_desc, rcl->d_before_bn_desc,
      CUDNN_BATCHNORM_PER_ACTIVATION));
  }

  /* set input & output tensor, space */
  float zero = 0;

  chkCUDNN(cudnnSetRNNDataDescriptor(rcl->input_rcnt_desc, CUDNN_DATA_FLOAT,
    CUDNN_RNN_DATA_LAYOUT_SEQ_MAJOR_UNPACKED, seq_length, batch_size,
    rcl->input_size, seqs, &zero));
  chkCUDNN(cudnnSetRNNDataDescriptor(rcl->after_rcnt_desc, CUDNN_DATA_FLOAT,
    CUDNN_RNN_DATA_LAYOUT_SEQ_MAJOR_UNPACKED, seq_length, batch_size,
    rcl->hidden_size * 2, seqs, &zero));
  chkCUDNN(cudnnSetRNNDataDescriptor(rcl->d_input_rcnt_desc, CUDNN_DATA_FLOAT,
    CUDNN_RNN_DATA_LAYOUT_SEQ_MAJOR_UNPACKED, seq_length, batch_size,
    rcl->input_size, seqs, &zero));
  chkCUDNN(cudnnSetRNNDataDescriptor(rcl->d_after_rcnt_desc, CUDNN_DATA_FLOAT,
    CUDNN_RNN_DATA_LAYOUT_SEQ_MAJOR_UNPACKED, seq_length, batch_size,
    rcl->hidden_size * 2, seqs, &zero));

  for (int i = 0; i < seq_length; i++) {
    rcl->cnt_seqs[i] = cnt_seqs[i];
    rcl->dim_input[0] = rcl->dim_output[0] = cnt_seqs[i];

    chkCUDNN(cudnnSetTensorNdDescriptor(rcl->_input_rcnt_desc[i],
      CUDNN_DATA_FLOAT, 3, rcl->dim_input, rcl->stride_input));
  }

  /* reduction */
  chkCUDNN(cudnnSetTensor4dDescriptor(rcl->a_desc, CUDNN_TENSOR_NCHW,
    CUDNN_DATA_FLOAT, batch_size * seq_length, 1, 2, rcl->hidden_size));
  chkCUDNN(cudnnSetTensor4dDescriptor(rcl->c_desc, CUDNN_TENSOR_NCHW,
    CUDNN_DATA_FLOAT, batch_size * seq_length, 1, 1, rcl->hidden_size));
  chkCUDNN(cudnnSetReduceTensorDescriptor(rcl->reduce_desc, CUDNN_REDUCE_TENSOR_ADD,
    CUDNN_DATA_FLOAT, CUDNN_NOT_PROPAGATE_NAN,
    CUDNN_REDUCE_TENSOR_NO_INDICES, CUDNN_32BIT_INDICES));

  chkCUDNN(cudnnGetReductionWorkspaceSize(rcl->cudnn, rcl->reduce_desc,
    rcl->a_desc, rcl->c_desc, &rcl->reduce_workspace_bytes));

  /* set hs output desc */
  chkCUDNN(cudnnSetTensorNdDescriptor(rcl->hs_output_desc, CUDNN_DATA_FLOAT, 3,
    rcl->dim_hidden_state, rcl->stride_hidden_state));

  chkCUDNN(cudnnSetTensorNdDescriptor(rcl->d_hs_output_desc, CUDNN_DATA_FLOAT, 3,
    rcl->dim_hidden_state, rcl->stride_hidden_state));
  
  /* allocate workspace */
  chkCUDNN(cudnnGetRNNWorkspaceSize(rcl->cudnn, rcl->rcnt_desc, seq_length,
    rcl->_input_rcnt_desc, &rcl->workspace_bytes));
  chkCUDNN(cudnnGetRNNTrainingReserveSize(rcl->cudnn, rcl->rcnt_desc,
    seq_length, rcl->_input_rcnt_desc, &rcl->reservespace_bytes));

  chkCUDA(cudaMalloc(&rcl->reservespace, rcl->reservespace_bytes));

  rcl->reduce_workspace = get_global_workspace(rcl->reduce_workspace_bytes);
  rcl->workspace = get_global_workspace(rcl->workspace_bytes);

  *output = rcl->summed_seq;
  if (rcl->bn_first == true) {
    *d_input = rcl->d_before_bn;
  } else {
    *d_input = rcl->d_input;
  }
}

void train_fwd_rcnt_layer(rcnt_layer *rcl)
{
  float one = 1.0; 
  float zero = 0.0;

  if (rcl->bn_first == true) {
    if (rcl->is_training == true) {
      START_STOPWATCH {
        chkCUDNN(cudnnBatchNormalizationForwardTraining(rcl->cudnn,
          CUDNN_BATCHNORM_PER_ACTIVATION, &one, &zero, rcl->before_bn_desc, rcl->before_bn,
          rcl->after_bn_desc, rcl->input, rcl->bn_desc, rcl->bn_scale, rcl->bn_bias,
          0.1, rcl->bn_result_running_mean, rcl->bn_result_running_var,
          1e-05, NULL, NULL));
          _DBG_SYNCHRONIZE();
      } STOP_STOPWATCH("  cudnn bn fwd training");
    } else {
      chkCUDNN(cudnnBatchNormalizationForwardInference(rcl->cudnn,
        CUDNN_BATCHNORM_PER_ACTIVATION, &one, &zero, rcl->before_bn_desc, rcl->before_bn,
        rcl->after_bn_desc, rcl->input, rcl->bn_desc, rcl->bn_scale, rcl->bn_bias,
        rcl->bn_result_running_mean, rcl->bn_result_running_var,
        1e-05));
    }
  }

  START_STOPWATCH {
    chkCUDNN(cudnnRNNForwardTrainingEx(rcl->cudnn, rcl->rcnt_desc,
      rcl->input_rcnt_desc, rcl->input, NULL, NULL, NULL, NULL,
      rcl->weight_desc, rcl->weight, rcl->after_rcnt_desc, rcl->after_rcnt,
      rcl->hs_output_desc, rcl->hs_output, NULL, NULL, NULL, NULL,
      NULL, NULL, NULL, NULL, NULL, NULL, *rcl->workspace, rcl->workspace_bytes,
      rcl->reservespace, rcl->reservespace_bytes));
    _DBG_SYNCHRONIZE();
  } STOP_STOPWATCH("  cudnn rnn forward");

  START_STOPWATCH {
    chkCUDNN(cudnnReduceTensor(rcl->cudnn, rcl->reduce_desc, NULL, 0,
      *rcl->reduce_workspace, rcl->reduce_workspace_bytes, &one,
      rcl->a_desc, rcl->after_rcnt, &zero, rcl->c_desc, rcl->summed_seq));
    _DBG_SYNCHRONIZE();
  } STOP_STOPWATCH("  cuda sum padded seq");
}

void train_bwd_rcnt_layer(rcnt_layer *rcl)
{
  assert(rcl->d_summed_seq != NULL); 

  START_STOPWATCH {
    deepspeech_cuda_expand_sum_padded_seq(rcl->d_summed_seq, rcl->d_padded_seq,
      rcl->hidden_size * 2, rcl->batch_size, rcl->seq_length);
  } STOP_STOPWATCH("  cuda expand");

  START_STOPWATCH {
    chkCUDNN(cudnnRNNBackwardDataEx(rcl->cudnn, rcl->rcnt_desc,
      rcl->after_rcnt_desc, rcl->after_rcnt, rcl->d_after_rcnt_desc, rcl->d_padded_seq,
      NULL, NULL, NULL, NULL, NULL, NULL, rcl->weight_desc, rcl->weight, NULL, NULL,
      NULL, NULL, rcl->d_input_rcnt_desc, rcl->d_input, NULL, NULL, NULL, NULL, NULL, NULL,
      *rcl->workspace, rcl->workspace_bytes, rcl->reservespace, rcl->reservespace_bytes));

    chkCUDNN(cudnnRNNBackwardWeightsEx(rcl->cudnn, rcl->rcnt_desc,
      rcl->input_rcnt_desc, rcl->input, NULL, NULL, rcl->after_rcnt_desc, rcl->after_rcnt,
      *rcl->workspace, rcl->workspace_bytes, rcl->d_weight_desc, rcl->d_weight,
      rcl->reservespace, rcl->reservespace_bytes));

    _DBG_SYNCHRONIZE();
  } STOP_STOPWATCH("  cudnn rnn backward");

  if (rcl->bn_first == true) {
    START_STOPWATCH {
      float one = 1.0;
      float zero = 0.0;
      chkCUDNN(cudnnBatchNormalizationBackward(rcl->cudnn,
        CUDNN_BATCHNORM_PER_ACTIVATION, &one, &zero, &one, &zero,
        rcl->before_bn_desc, rcl->before_bn,
        rcl->d_after_bn_desc, rcl->d_input,
        rcl->d_before_bn_desc, rcl->d_before_bn,
        rcl->d_bn_desc, rcl->bn_scale, rcl->d_bn_scale, rcl->d_bn_bias,
        1e-05, NULL, NULL));
      _DBG_SYNCHRONIZE();
    } STOP_STOPWATCH("  cudnn batchnorm backward");
  }
}

void free_rcnt_layer(rcnt_layer *rcl)
{
  chkCUDA(cudaFree(rcl->reservespace));
}

void set_rcnt_layer_params(rcnt_layer *rcl, float *weights, float *biases)
{
  cudnnFilterDescriptor_t lin_layer_param_desc;
  size_t w_matrix_size = rcl->hidden_size * rcl->input_size;
  size_t r_matrix_size = rcl->hidden_size * rcl->hidden_size;
  size_t bias_size = rcl->hidden_size;
  int pl, l;

  float *cursor_w = weights;
  float *cursor_b = biases;
  void *w_ptr, *b_ptr;

  chkCUDNN(cudnnCreateFilterDescriptor(&lin_layer_param_desc));

  for (int pl = 0; pl < 2; pl++) {
    for (int l = 0; l < 6; l++) {
      size_t matrix_size = (l < 3 ? w_matrix_size : r_matrix_size);

      chkCUDNN(cudnnGetRNNLinLayerMatrixParams(rcl->cudnn,
        rcl->rcnt_desc, pl, rcl->x, rcl->weight_desc, rcl->weight, 
        l, lin_layer_param_desc, &w_ptr));
      chkCUDNN(cudnnGetRNNLinLayerBiasParams(rcl->cudnn,
        rcl->rcnt_desc, pl, rcl->x, rcl->weight_desc, rcl->weight, 
        l, lin_layer_param_desc, &b_ptr));

      chkCUDA(cudaMemcpy(w_ptr, cursor_w, sizeof(float) * matrix_size,
        cudaMemcpyHostToDevice));
      chkCUDA(cudaMemcpy(b_ptr, cursor_b, sizeof(float) * bias_size,
        cudaMemcpyHostToDevice));

      cursor_w += matrix_size;
      cursor_b += bias_size;
    }
  }
}

