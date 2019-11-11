#include <stdio.h>
#include <string.h>
#include <stdbool.h>
#include <assert.h>
#include <math.h>
#include <time.h>

#include <cuda.h>
#include <cuda_profiler_api.h>
#include <cudnn.h>
#include <cublas_v2.h>

#include <mpi.h>
#include <nccl.h>

#include "deepspeech.h"
#include "deepspeech_cuda.h"

#include "params.h"
#include "utils.h"

#include "fc.h"
#include "conv.h"
#include "rcnt.h"

#include "wer.hpp"

#include "ctc.h"

const float one = 1; // TODO
const float zero = 0;

cudnnHandle_t cudnn;
cublasHandle_t cublas;

cudaStream_t cudnn_stream;

cudnnCTCLossDescriptor_t ctc_loss_desc;
cudnnTensorDescriptor_t grads_desc;
cudnnTensorDescriptor_t probs_desc;

float *p;
float *d;
float *buf_d = NULL;

int off_d;

int rcnt_input_size;

int *batch_indices;
int *labels;
int *label_lengths;
int *input_lengths;

float *_input;

float *input;
float *cvl_output_1;
float *cvl_output_2;
float *rcnt_input;
float *rcnt_output;
float *fcl_output;
float *output;

float *grads;
float *d_cvl_output_2;
float *d_rcnt_input;
float *d_input;
float *loss;
void **ctc_workspace;

conv_layer cvl_1;
conv_layer cvl_2;
rcnt_layer rcl[5];
fc_layer fcl;

bool is_initiated = false;

float *tmp_conv_out;

/* inital values */
float *conv1_weight, *conv1_bn_scale, *conv1_bn_bias;
float *conv2_weight, *conv2_bn_scale, *conv2_bn_bias;
float *rcnt_weight[5], *rcnt_bias[5];
float *rcnt_bn_scale[5], *rcnt_bn_bias[5];
float *fc_filter, *fc_bn_scale, *fc_bn_bias;

void deepspeech_init(int batch_size, int max_seq)
{
  off_d = 0;

  rcnt_input_size =
    32 * CALC_SIZE(CALC_SIZE(FIXED_HEIGHT, 41, 0, 2), 21, 0, 2);
  int _max_seq = CALC_SIZE(CALC_SIZE(max_seq, 11, 10, 2), 11, 0, 1);

  chkCUDNN(cudnnCreate(&cudnn));
  chkCUBLAS(cublasCreate(&cublas));

  chkCUDNN(cudnnGetStream(cudnn, &cudnn_stream));

  chkCUDNN(cudnnCreateCTCLossDescriptor(&ctc_loss_desc));
  chkCUDNN(cudnnCreateTensorDescriptor(&grads_desc));
  chkCUDNN(cudnnCreateTensorDescriptor(&probs_desc));
  chkCUDNN(cudnnSetCTCLossDescriptor(ctc_loss_desc,
    CUDNN_DATA_FLOAT));

  init_conv_layer(&cvl_1, cudnn, 41, 11, 0, 10, 2, 2, 32, 1,
    conv1_weight, conv1_bn_scale, conv1_bn_bias, &off_d,
    batch_size, FIXED_HEIGHT, max_seq);
  init_conv_layer(&cvl_2, cudnn, 21, 11, 0, 0, 2, 1, 32, 32,
    conv2_weight, conv2_bn_scale, conv2_bn_bias, &off_d,
    batch_size, CALC_SIZE(FIXED_HEIGHT, 41, 0, 2),
    CALC_SIZE(max_seq, 11, 10, 2));

  init_rcnt_layer(&rcl[0], cudnn, CUDNN_GRU, rcnt_input_size,
    params.rnn_hidden_size, batch_size, _max_seq, false,
    rcnt_weight[0], rcnt_bias[0], rcnt_bn_scale[0], rcnt_bn_bias[0], &off_d);
  init_rcnt_layer(&rcl[1], cudnn, CUDNN_GRU, params.rnn_hidden_size,
    params.rnn_hidden_size, batch_size, _max_seq, true,
    rcnt_weight[1], rcnt_bias[1], rcnt_bn_scale[1], rcnt_bn_bias[1], &off_d);
  init_rcnt_layer(&rcl[2], cudnn, CUDNN_GRU, params.rnn_hidden_size,
    params.rnn_hidden_size, batch_size, _max_seq, true,
    rcnt_weight[2], rcnt_bias[2], rcnt_bn_scale[2], rcnt_bn_bias[2], &off_d);
  init_rcnt_layer(&rcl[3], cudnn, CUDNN_GRU, params.rnn_hidden_size,
    params.rnn_hidden_size, batch_size, _max_seq, true,
    rcnt_weight[3], rcnt_bias[3], rcnt_bn_scale[3], rcnt_bn_bias[3], &off_d);
  init_rcnt_layer(&rcl[4], cudnn, CUDNN_GRU, params.rnn_hidden_size,
    params.rnn_hidden_size, batch_size, _max_seq, true,
    rcnt_weight[4], rcnt_bias[4], rcnt_bn_scale[4], rcnt_bn_bias[4], &off_d);

  init_fc_layer(&fcl, cudnn, params.rnn_hidden_size, 29,
    fc_filter, fc_bn_scale, fc_bn_bias, &off_d, batch_size * _max_seq);

  is_initiated = true;

  chkCUDA(cudaMalloc((void **)&d, sizeof(float) * off_d));
  chkCUDA(cudaMalloc((void **)&p, sizeof(float) * off_d));
  chkCUDA(cudaMemset(p, 0, sizeof(float) * off_d));

  fprintf(stderr, "num_params : %d\n", off_d);
}

void deepspeech_sort_batch(dataset_t *dataset, int batch_size)
{
  int tmp;

  for (int i = 0; i < batch_size; i++) {
    for (int j = i + 1; j < batch_size; j++) {
      if (dataset->widths_wav[batch_indices[i]] <
        dataset->widths_wav[batch_indices[j]]) {

        tmp = batch_indices[i];
        batch_indices[i] = batch_indices[j];
        batch_indices[j] = tmp;
      }
    }
  }
}

void deepspeech_load_inputs(dataset_t *dataset, int batch_size, int max_width)
{
  for (int i = 0; i < batch_size; i++) {
    dataset_get_wav(dataset, batch_indices[i], _input + i * max_width * FIXED_HEIGHT);
  } 
}

float *tmp;

void deepspeech_set_tensors(dataset_t *dataset, int batch_size,
  int max_width, bool is_training)
{
  chkCUDA(cudaMemset(d, 0, sizeof(float) * off_d));

  int _max_width = CALC_SIZE(CALC_SIZE(max_width, 11, 10, 2), 11, 0, 1);
  int *cnt_seqs = (int *)malloc(sizeof(int) * _max_width);
  int *seqs = (int *)malloc(sizeof(int) * batch_size);

  MALLOC_TENSOR_FLOATZ(&input, batch_size, 1, FIXED_HEIGHT, max_width);
  MALLOC_TENSOR_FLOATZ(&rcnt_input, batch_size, rcnt_input_size, _max_width, 1);
  MALLOC_TENSOR_FLOATZ(&grads, _max_width, batch_size, 1, 29);
  MALLOC_TENSOR_FLOAT(&d_cvl_output_2, batch_size,
    1, rcnt_input_size, _max_width);

  for (int j = 0; j < batch_size; j++) {
    int width = dataset->widths_wav[batch_indices[j]];
    if (width % 2 == 0) width++;

    int _width = CALC_SIZE(CALC_SIZE(width, 11, 10, 2), 11, 0, 1);
    seqs[j] = _width;
  }

  for (int i = 0; i < _max_width; i++) {
    int n = batch_size;
    for (int j = 0; j < batch_size; j++) {
      int width = dataset->widths_wav[batch_indices[j]];
      if (width % 2 == 0) width++;

      int _width = CALC_SIZE(CALC_SIZE(width, 11, 10, 2), 11, 0, 1);
      if (_width <= i) {
        n--;
      }
    }
    cnt_seqs[i] = n;
  }

  int dims[3] = {_max_width, batch_size, 29};
  int strides[3] = {batch_size * 29, 29, 1};

  chkCUDNN(cudnnSetTensorNdDescriptor(probs_desc, CUDNN_DATA_FLOAT, 3,
    dims, strides));
  chkCUDNN(cudnnSetTensorNdDescriptor(grads_desc, CUDNN_DATA_FLOAT, 3,
    dims, strides));

  set_conv_layer(&cvl_1, input, &cvl_output_1,
    NULL, &d_input, batch_size, FIXED_HEIGHT, max_width, is_training);
  set_conv_layer(&cvl_2, cvl_output_1, &cvl_output_2,
    d_cvl_output_2, &cvl_1.d_after_act,
    batch_size, cvl_1.output_height, cvl_1.output_width, is_training);

  set_rcnt_layer(&rcl[0], rcnt_input, &rcl[1].input,
    NULL, &d_rcnt_input,
    batch_size, cvl_2.output_width, cnt_seqs, seqs, is_training);
  set_rcnt_layer(&rcl[1], rcl[0].output, &rcl[2].input,
    NULL, &rcl[0].d_summed_seq,
    batch_size, cvl_2.output_width, cnt_seqs, seqs, is_training);
  set_rcnt_layer(&rcl[2], rcl[1].output, &rcl[3].input,
    NULL, &rcl[1].d_summed_seq,
    batch_size, cvl_2.output_width, cnt_seqs, seqs, is_training);
  set_rcnt_layer(&rcl[3], rcl[2].output, &rcl[4].input,
    NULL, &rcl[2].d_summed_seq,
    batch_size, cvl_2.output_width, cnt_seqs, seqs, is_training);
  set_rcnt_layer(&rcl[4], rcl[3].output, &rcnt_output,
    NULL, &rcl[3].d_summed_seq,
    batch_size, cvl_2.output_width, cnt_seqs, seqs, is_training);

  set_fc_layer(&fcl, rcnt_output, &fcl_output, grads, &rcl[4].d_summed_seq,
    batch_size * _max_width, is_training);

  free(cnt_seqs);
  free(seqs);
}

void deepspeech_copy_inputs(dataset_t *dataset, int batch_size, int max_width)
{
  void *dst_cur = input;
  void *src_cur = _input;

  for (int i = 0; i < batch_size; i++) {
    int width = dataset->widths_wav[batch_indices[i]];
    for (int j = 0; j < FIXED_HEIGHT; j++) {
      chkCUDNN(cudaMemcpy(dst_cur, src_cur,
        sizeof(float) * width, cudaMemcpyHostToDevice));
      dst_cur += sizeof(float) * max_width;
      src_cur += sizeof(float) * width;
    }
  }
}

void deepspeech_forward(int batch_size, int max_width)
{
  int _max_width = CALC_SIZE(CALC_SIZE(max_width, 11, 10, 2), 11, 0, 1);

  START_STOPWATCH {
    train_fwd_conv_layer(&cvl_1);
    _DBG_SYNCHRONIZE();
  } STOP_STOPWATCH("train fwd conv layer 1");
  START_STOPWATCH {
    train_fwd_conv_layer(&cvl_2);
    _DBG_SYNCHRONIZE();
  } STOP_STOPWATCH("train fwd conv layer 2");

  START_STOPWATCH {
    deepspeech_cuda_transpose(cvl_output_2, rcnt_input, batch_size,
      cvl_2.channels_out * cvl_2.output_height, cvl_2.output_width);
    _DBG_SYNCHRONIZE();
  } STOP_STOPWATCH("deepspeech cuda transpose");

  START_STOPWATCH {
    train_fwd_rcnt_layer(&rcl[0]);
  } STOP_STOPWATCH("train fwd rcnt layer 0");
  START_STOPWATCH {
    train_fwd_rcnt_layer(&rcl[1]);
  } STOP_STOPWATCH("train fwd rcnt layer 1");
  START_STOPWATCH {
    train_fwd_rcnt_layer(&rcl[2]);
  } STOP_STOPWATCH("train fwd rcnt layer 2");
  START_STOPWATCH {
    train_fwd_rcnt_layer(&rcl[3]);
  } STOP_STOPWATCH("train fwd rcnt layer 3");
  START_STOPWATCH {
    train_fwd_rcnt_layer(&rcl[4]);
  } STOP_STOPWATCH("train fwd rcnt layer 4");
  START_STOPWATCH {
    train_fwd_fc_layer(&fcl);
  } STOP_STOPWATCH("train fwd fc layer");
}

float deepspeech_calc_loss(int batch_size)
{
  struct ctcOptions options;
  size_t workspace_size;
  float *_loss, ret = 0;

  options.loc = CTC_GPU;
  options.stream = cudnn_stream;
  options.blank_label = 0;

  _loss = (float *)malloc(sizeof(float) * batch_size);

  CTC_CALL(get_workspace_size(label_lengths, input_lengths,
    NUM_LABEL, batch_size, options, &workspace_size));
  ctc_workspace = get_global_workspace(workspace_size);

  CTC_CALL(compute_ctc_loss(fcl_output, grads,
    labels, label_lengths, input_lengths, NUM_LABEL,
    batch_size, _loss, *ctc_workspace, options));

  for (int b = 0; b < batch_size; b++) ret += _loss[b];
  ret /= batch_size;

  free(_loss);

  return ret;
}

void deepspeech_backward(int batch_size)
{
  START_STOPWATCH {
    train_bwd_fc_layer(&fcl);
    _DBG_SYNCHRONIZE();
  } STOP_STOPWATCH("train bwd fc layer");
  START_STOPWATCH {
    train_bwd_rcnt_layer(&rcl[4]);
  } STOP_STOPWATCH("train bwd rcnt 4 layer");
  START_STOPWATCH {
    train_bwd_rcnt_layer(&rcl[3]);
  } STOP_STOPWATCH("train bwd rcnt 3 layer");
  START_STOPWATCH {
    train_bwd_rcnt_layer(&rcl[2]);
  } STOP_STOPWATCH("train bwd rcnt 2 layer");
  START_STOPWATCH {
    train_bwd_rcnt_layer(&rcl[1]);
  } STOP_STOPWATCH("train bwd rcnt 1 layer");
  START_STOPWATCH {
    train_bwd_rcnt_layer(&rcl[0]);
  } STOP_STOPWATCH("train bwd rcnt 0 layer");

  START_STOPWATCH {
    deepspeech_cuda_transpose_inverse(d_rcnt_input, d_cvl_output_2, batch_size,
      cvl_2.channels_out * cvl_2.output_height, cvl_2.output_width);
    _DBG_SYNCHRONIZE();
  } STOP_STOPWATCH("deepspeech cuda transpose inverse");

  START_STOPWATCH {
    train_bwd_conv_layer(&cvl_2);
    _DBG_SYNCHRONIZE();
  } STOP_STOPWATCH("train bwd conv layer 2");
  START_STOPWATCH {
    train_bwd_conv_layer(&cvl_1);
    _DBG_SYNCHRONIZE();
  } STOP_STOPWATCH("train bwd conv layer 1");
}
void deepspeech_update(float lr, float grads_sum)
{
  float clip = grads_sum / params.max_norm;
  if (clip < 1.0f) {
    clip = 1.0f;
  }

  deepspeech_cuda_apply_grad(p, d, &buf_d, lr, clip, 1, 1, 1, off_d);   
}

void deepspeech_free(bool is_training)
{
  chkCUDA(cudaFree(input));
  chkCUDA(cudaFree(rcnt_input));
  chkCUDA(cudaFree(grads));

  chkCUDA(cudaFree(d_cvl_output_2));

  free_conv_layer(&cvl_1);
  free_conv_layer(&cvl_2);
  free_rcnt_layer(&rcl[0]);
  free_rcnt_layer(&rcl[1]);
  free_rcnt_layer(&rcl[2]);
  free_rcnt_layer(&rcl[3]);
  free_rcnt_layer(&rcl[4]);
  free_fc_layer(&fcl);
}


float deepspeech_calc_wer(dataset_t *dataset, int batch_size,
  int max_width, int max_target_length)
{
  int _max_width = CALC_SIZE(CALC_SIZE(max_width, 11, 10, 2), 11, 0, 1);
  size_t total_bytes = sizeof(float) * batch_size * _max_width * NUM_LABEL;
  float wer = 0;
  float *probs = (float *)malloc(total_bytes); // T * N * H
  char *chars = (char *)malloc(sizeof(char) * _max_width + 1);
  char *target = (char *)malloc(sizeof(char) * max_target_length + 1);
  int *txts = (int *)malloc(sizeof(int) * max_target_length + 1);

  chkCUDA(cudaMemcpy(probs, fcl_output, total_bytes, cudaMemcpyDeviceToHost));

  for (int n = 0; n < batch_size; n++) {
    int target_width = dataset->widths_txt[batch_indices[n]];
    int len_chars = 0;
    int len_target = 0;
    for (int w = 0; w < _max_width; w++) {
      float max = 0;
      int max_idx = -1;
      for (int c = 0; c < NUM_LABEL; c++) {
        if (max_idx == -1 ||
          max < probs[w * batch_size * NUM_LABEL + n * NUM_LABEL + c]) {
          max = probs[w * batch_size * NUM_LABEL + n * NUM_LABEL + c];
          max_idx = c;
        }
      }

      if (max_idx != 0 && !(len_chars > 0 &&
        char_table[max_idx] == chars[len_chars - 1])) {
        chars[len_chars++] = char_table[max_idx];
      }
    }

    dataset_get_txt(dataset, batch_indices[n], txts);

    for (int w = 0; w < target_width; w++) {
      if (txts[w] == 0 && !(len_target > 0 &&
        char_table[txts[w]] == target[len_target - 1])) {
        continue;
      }
      target[len_target++] = char_table[txts[w]];
    }

    chars[len_chars] = target[len_target] = 0;
    wer += calc_wer(chars, target, len_chars, len_target);
  }

  free(txts);
  free(target);
  free(chars);
  free(probs);

  return wer;
}

void deepspeech_eval(dataset_t *dataset_val, int world_rank,
  int world_size, int num_dev_per_node)
{
  int max_target_length = 0;
  int max_seq = 0;
  int num_samples = dataset_val->size;
  int num_batches = CDIV(num_samples, params.batch_size_eval * world_size);
  int rest = num_samples;
  int *indices = (int *)malloc(sizeof(int) * num_samples);

  double total_time = 0;

  ncclUniqueId nccl_id;
  ncclComm_t nccl_comm;

  chkCUDA(cudaSetDevice(world_rank % num_dev_per_node));

  batch_indices = (int *)malloc(sizeof(int) * params.batch_size_eval);
  indices = (int *)malloc(sizeof(int) * num_samples);
  for (int i = 0; i < num_samples; i++) indices[i] = i;

  int sum_seq = 0;

  for (int i = 0; i < num_samples; i++) {
    int width_txt = dataset_val->widths_txt[i];
    max_target_length = MAX(width_txt, max_target_length);
  }

  for (int i = 0; i < num_samples; i++) {
    int width_wav = dataset_val->widths_wav[i];
    max_seq = MAX(width_wav, max_seq);
    sum_seq += width_wav;
  }

  if (max_seq % 2 == 0) max_seq++;

  _input = (float *)malloc(sizeof(float) * params.batch_size_eval *
    max_seq * FIXED_HEIGHT);

  deepspeech_init(params.batch_size_eval, max_seq);

  if (world_rank == 0) NCCL_CALL(ncclGetUniqueId(&nccl_id));
  MPI_CALL(MPI_Bcast(&nccl_id, sizeof(nccl_id), MPI_BYTE, 0, MPI_COMM_WORLD));
  NCCL_CALL(ncclCommInitRank(&nccl_comm, world_size, nccl_id, world_rank));

  NCCL_CALL(ncclBcast(p, off_d, ncclFloat32, 0, nccl_comm, cudnn_stream));
  chkCUDA(cudaStreamSynchronize(cudnn_stream));

  float total_wer = 0.0;

  for (int b = 0; b < num_batches; b++) {
    int batch_size_total = 0;
    int batch_size = 0;
    int batch_offset;
    int max_width;
    int max_target_length_ = 0;

    double start_time;
    double elapsed_time;

    if (rest >= params.batch_size_eval * world_size) {
      batch_size = params.batch_size_eval;
      batch_size_total = params.batch_size_eval * world_size;
      batch_offset = b * params.batch_size_eval * world_size +
        params.batch_size_eval * world_rank;
    } else {
      batch_size = rest / world_size;
      if (world_rank < rest % world_size) {
        batch_size += 1;
      }

      batch_size_total = rest;
      batch_offset = b * params.batch_size_eval * world_size +
        (rest / world_size) * world_rank + MIN(rest % world_size, world_rank);
    }

    START_STOPWATCH {
      if (batch_size > 0) {
        for (int s = 0; s < batch_size; s++) {
          batch_indices[s] = indices[batch_offset + s];
          max_target_length_ = MAX(max_target_length_,
            dataset_val->widths_txt[batch_indices[s]]);
        }

        deepspeech_sort_batch(dataset_val, batch_size);
        max_width = dataset_val->widths_wav[batch_indices[0]];
        if (max_width % 2 == 0) {
          max_width++;
        }

        deepspeech_load_inputs(dataset_val, batch_size, max_width);

        START_STOPWATCH {
          deepspeech_set_tensors(dataset_val, batch_size, max_width, false);
        } STOP_STOPWATCH("deepspeech set tensors");

        START_STOPWATCH {
          deepspeech_copy_inputs(dataset_val, batch_size, max_width);
        } STOP_STOPWATCH("deepspeech copy inputs");
      }

      chkCUDA(cudaDeviceSynchronize());
      MPI_Barrier(MPI_COMM_WORLD);
      start_time = MPI_Wtime();
      {
        if (batch_size > 0) {
          deepspeech_forward(batch_size, max_width);
        }
      }
      chkCUDA(cudaDeviceSynchronize());
      MPI_Barrier(MPI_COMM_WORLD);

      elapsed_time = MPI_Wtime() - start_time;
      total_time += elapsed_time;

      total_wer += deepspeech_calc_wer(dataset_val, batch_size,
        max_width, max_target_length_);

      if (batch_size > 0) {
        START_STOPWATCH {
          deepspeech_free(false);
        } STOP_STOPWATCH("deepspeech free");
      }
    } STOP_STOPWATCH("total");


    if (world_rank == 0) {
      fprintf(stderr, "batch : %d/%d, elapsed time : %lf, "
        "total time : %lf, max_width : %d\n", b, num_batches - 1,
        elapsed_time, total_time, max_width);
    }

    rest -= batch_size_total;
  }

  float final_wer = 0;
  MPI_CALL(MPI_Reduce(&total_wer, &final_wer, 1, MPI_FLOAT, MPI_SUM, 0, MPI_COMM_WORLD));
  if (world_rank == 0) {
    fprintf(stderr, "final wer : %f\n", final_wer / num_samples);
  }
  free(indices);
}

void deepspeech_train(dataset_t *dataset_train, int world_rank,
  int world_size, int num_dev_per_node)
{
  float learning_rate = params.learning_rate;
  int max_target_length = 0;
  int max_seq = 0;
  int num_samples = dataset_train->size;
  int num_batches = CDIV(num_samples, params.batch_size * world_size);
  int rest = num_samples;
  int *indices;

  double total_time = 0;

  ncclUniqueId nccl_id;
  ncclComm_t nccl_comm;

  chkCUDA(cudaSetDevice(world_rank % num_dev_per_node));

  batch_indices = (int *)malloc(sizeof(int) * params.batch_size);
  indices = (int *)malloc(sizeof(int) * num_samples);
  for (int i = 0; i < num_samples; i++) indices[i] = i;

  int total_length = 0;
  for (int i = 0; i < num_samples; i++) {
    int width_txt = dataset_train->widths_txt[i];
    max_target_length = MAX(width_txt, max_target_length);
    total_length += width_txt;
  }

  int total_seq = 0;
  for (int i = 0; i < num_samples; i++) {
    int width_wav = dataset_train->widths_wav[i];
    max_seq = MAX(width_wav, max_seq);
    total_seq += width_wav;
  }

  if (max_seq % 2 == 0) max_seq++;

  _input = (float *)malloc(sizeof(float) * params.batch_size *
    max_seq * FIXED_HEIGHT);
  labels = (int *)malloc(sizeof(int) * params.batch_size * max_target_length);
  label_lengths = (int *)malloc(sizeof(int) * params.batch_size);
  input_lengths = (int *)malloc(sizeof(int) * params.batch_size);

  deepspeech_init(params.batch_size, max_seq);

  if (world_rank == 0) NCCL_CALL(ncclGetUniqueId(&nccl_id));
  MPI_CALL(MPI_Bcast(&nccl_id, sizeof(nccl_id), MPI_BYTE, 0, MPI_COMM_WORLD));
  NCCL_CALL(ncclCommInitRank(&nccl_comm, world_size, nccl_id, world_rank));

  NCCL_CALL(ncclBcast(p, off_d, ncclFloat32, 0, nccl_comm, cudnn_stream));
  chkCUDA(cudaStreamSynchronize(cudnn_stream));

  float loss;

  void **tmp = get_global_workspace(PRE_ALLOC_SIZE);

  int tcnt = 0;
  for (int e = 0; e < params.epochs; e++) {
    rest = num_samples;
    if (world_rank == 0) fprintf(stderr, "epoch %d start, %d batches\n", e, num_batches);

    for (int b = 0; b < num_batches; b++, tcnt++) {
      int batch_size_total = 0;
      int batch_size = 0;
      int batch_offset;
      int max_width;
      int max_target_length_ = 0;
      float grads_sum = 0;
      float loss_sum;
      float wer, wer_sum;
      double start_time;
      double elapsed_time;
      size_t free_byte, total_byte;

      if (rest >= params.batch_size * world_size) {
        batch_size = params.batch_size;
        batch_size_total = params.batch_size * world_size;
        batch_offset = b * params.batch_size * world_size +
          params.batch_size * world_rank;
      } else {
        batch_size = rest / world_size;
        if (world_rank < rest % world_size) {
          batch_size += 1;
        }

        batch_size_total = rest;
        batch_offset = b * params.batch_size * world_size +
          (rest / world_size) * world_rank + MIN(rest % world_size, world_rank);
      }

      START_STOPWATCH {
        if (batch_size > 0) {
          for (int s = 0; s < batch_size; s++) {
            batch_indices[s] = indices[batch_offset + s];
            max_target_length_ = MAX(max_target_length_,
              dataset_train->widths_txt[batch_indices[s]]);
          }

          deepspeech_sort_batch(dataset_train, batch_size);
          max_width = dataset_train->widths_wav[batch_indices[0]];
          if (max_width % 2 == 0) {
            max_width++;
          }

          int cur = 0;
          for (int s = 0; s < batch_size; s++) {
            label_lengths[s] = dataset_train->widths_txt[batch_indices[s]];
            input_lengths[s] = CALC_SIZE(CALC_SIZE(max_width,
              11, 10, 2), 11, 0, 1);
            dataset_get_txt(dataset_train, batch_indices[s], labels + cur);
            cur += label_lengths[s];
          }

          deepspeech_load_inputs(dataset_train, batch_size, max_width);

          START_STOPWATCH {
            deepspeech_set_tensors(dataset_train, batch_size, max_width, true);
          } STOP_STOPWATCH("deepspeech set tensors");

          START_STOPWATCH {
            deepspeech_copy_inputs(dataset_train, batch_size, max_width);
          } STOP_STOPWATCH("deepspeech copy inputs");
        }

        MPI_Barrier(MPI_COMM_WORLD);
        start_time = MPI_Wtime();
        {
          if (batch_size > 0) {
            START_STOPWATCH {
              deepspeech_forward(batch_size, max_width);
            } STOP_STOPWATCH("deepspeech forward");

            START_STOPWATCH {
              loss = deepspeech_calc_loss(batch_size);
            } STOP_STOPWATCH("deepspeech calc loss");

            START_STOPWATCH {
              deepspeech_backward(batch_size);
            } STOP_STOPWATCH("deepspeech backward");
          }

          NCCL_CALL(ncclAllReduce(d, d, off_d, ncclFloat32,
            ncclSum, nccl_comm, cudnn_stream));
          chkCUDA(cudaStreamSynchronize(cudnn_stream));

          START_STOPWATCH {
            grads_sum = deepspeech_cuda_sum_square(d, 1, 1, 1, off_d);
            _DBG_SYNCHRONIZE();
          } STOP_STOPWATCH("calc grads sum");

          START_STOPWATCH {
            deepspeech_update(learning_rate, grads_sum);
          } STOP_STOPWATCH("deepspeech update");

          loss *= batch_size;
          MPI_CALL(MPI_Reduce(&loss, &loss_sum, 1, MPI_FLOAT, MPI_SUM, 0, MPI_COMM_WORLD));
        }
        MPI_Barrier(MPI_COMM_WORLD);

        elapsed_time = MPI_Wtime() - start_time;
        total_time += elapsed_time;

      } STOP_STOPWATCH("total");

      wer = deepspeech_calc_wer(dataset_train, batch_size,
      max_width, max_target_length_);

      MPI_CALL(MPI_Reduce(&wer, &wer_sum, 1, MPI_FLOAT, MPI_SUM, 0, MPI_COMM_WORLD));

      if (batch_size > 0) {
        START_STOPWATCH {
          deepspeech_free(true);
        } STOP_STOPWATCH("deepspeech free");
      }

      if (world_rank == 0) {
        fprintf(stderr, "batch : %d/%d, grads_sum : %f, loss : %f, "
          "elapsed time : %lf, total time : %lf, max_width : %d, WER : %f\n",
          b, num_batches - 1, grads_sum, loss_sum / batch_size_total,
          elapsed_time, total_time, max_width, wer / batch_size_total);
      }

      rest -= batch_size_total;
    }

    learning_rate /= params.learning_anneal;
  }

  free(labels);
  free(label_lengths);
  free(input_lengths);

  NCCL_CALL(ncclCommDestroy(nccl_comm));
}
