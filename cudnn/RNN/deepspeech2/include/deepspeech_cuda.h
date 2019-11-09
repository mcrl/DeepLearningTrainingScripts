#ifndef _DEEPSPEECH_CUDA_H_
#define _DEEPSPEECH_CUDA_H_

#ifdef __cplusplus
#define EXTERNC extern "C"
#else
#define EXTERNC
#endif

EXTERNC void deepspeech_cuda_transpose(float *in, float *out, int N, int H, int W);
EXTERNC void deepspeech_cuda_transpose_inverse(float *in, float *out,
  int N, int H, int W);
EXTERNC void deepspeech_cuda_pack_padded_seq(float *in, float *out,
  int input_size, int batch_size, int seq_length, int *cnt_seqs);
EXTERNC void deepspeech_cuda_pad_packed_seq(float *in, float *out,
  int input_size, int batch_size, int seq_length, int *cnt_seqs);
EXTERNC void deepspeech_cuda_sum_padded_seq(float *in, float *out,
  int input_size, int batch_size, int seq_length);
EXTERNC void deepspeech_cuda_apply_grad(float *x, float *dx, float **buf,
  float lr, float clip, int N, int C, int H, int W);
EXTERNC void deepspeech_cuda_expand_sum_padded_seq(float *in, float *out,
  int input_size, int batch_size, int seq_length);
EXTERNC float deepspeech_cuda_sum_square(float *t, int N, int C, int H, int W);

#endif // _DEEPSPEECH_CUDA_H_

