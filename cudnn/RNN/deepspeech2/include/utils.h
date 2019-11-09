#ifndef _UTILS_H_
#define _UTILS_H_

#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <time.h>

#include <cuda.h>
#include <cudnn.h>
#include <cublas_v2.h>

#define CDIV(a, b) (((a) + (b) - 1) / (b))
#define MAX(a, b) (((a) > (b)) ? (a) : (b))
#define MIN(a, b) (((a) < (b)) ? (a) : (b))

#define START_STOPWATCH_EX                                     \
  {                                                            \
    _DBG_SYNCHRONIZE();                                        \
    MPI_Barrier(MPI_COMM_WORLD);                               \
    double st;                                                 \
    st = MPI_Wtime();

#define STOP_STOPWATCH_EX(tg)                                  \
    _DBG_SYNCHRONIZE();                                        \
    MPI_Barrier(MPI_COMM_WORLD);                               \
    double elapsed = MPI_Wtime() - st;                         \
    fprintf(stderr,"%s: %fms\n", tg, elapsed);                 \
  }


#if defined _DEBUG_ && defined MPI_VERSION
#define START_STOPWATCH                                        \
  {                                                            \
    _DBG_SYNCHRONIZE();                                        \
    MPI_Barrier(MPI_COMM_WORLD);                               \
    double st;                                                 \
    st = MPI_Wtime();

#define STOP_STOPWATCH(tg)                                     \
    _DBG_SYNCHRONIZE();                                        \
    MPI_Barrier(MPI_COMM_WORLD);                               \
    double elapsed = MPI_Wtime() - st;                         \
    fprintf(stderr,"%s: %fms\n", tg, elapsed);                 \
  }
#else
#define START_STOPWATCH
#define STOP_STOPWATCH(tg)
#endif

#define _DBG_SYNCHRONIZE() chkCUDA(cudaDeviceSynchronize())

#define CTC_CALL(exp)                                          \
  {                                                            \
    int status = (exp);                                        \
    if (status != CTC_STATUS_SUCCESS) {                        \
      fprintf(stderr, "[%s] Error on line %d: %s\n",           \
        __FILE__, __LINE__, ctcGetStatusString(status));       \
      exit(99);                                                \
    }                                                          \
  }

#define MPI_CALL(exp)                                          \
  {                                                            \
    int status = (exp);                                        \
    if (status != MPI_SUCCESS) {                               \
      fprintf(stderr, "[%s] Error on line %d: %s\n",           \
        __FILE__, __LINE__, "mpi error");                      \
      exit(99);                                                \
    }                                                          \
  }

#define NCCL_CALL(exp)                                         \
  {                                                            \
    ncclResult_t status = (exp);                               \
    if (status != ncclSuccess) {                               \
      fprintf(stderr, "[%s] Error on line %d: %s\n",           \
        __FILE__, __LINE__, "nccl error");                     \
      exit(99);                                                \
    }                                                          \
  }

#define chkCUDA(exp)                                           \
  {                                                            \
    cudaError_t status = (exp);                                \
    if (status != cudaSuccess) {                               \
      fprintf(stderr, "[%s] Error on line %d: %s\n",           \
        __FILE__, __LINE__, cudaGetErrorString(status));       \
      exit(99);                                                \
    }                                                          \
  }

#define chkCUDNN(exp)                                          \
  {                                                            \
    cudnnStatus_t status = (exp);                              \
    if (status != CUDNN_STATUS_SUCCESS) {                      \
      fprintf(stderr, "[%s] Error on line %d: %s\n",           \
        __FILE__, __LINE__, cudnnGetErrorString(status));      \
      exit(99);                                                \
    }                                                          \
  }

#define chkCUBLAS(exp)                                         \
  {                                                            \
    cublasStatus_t status = (exp);                              \
    if (status != CUBLAS_STATUS_SUCCESS) {                     \
      fprintf(stderr, "[%s] Error on line %d\n",               \
        __FILE__, __LINE__);                                   \
      exit(99);                                                \
    }                                                          \
  }

/*
static inline float diff_timespec_ms(struct timespec st, struct timespec ed)
{
  return (ed.tv_sec - st.tv_sec) * 1000 +
    (ed.tv_nsec - st.tv_nsec) / 1000000.0;
}
*/

static inline void MALLOC_TENSOR_FLOAT(float **pptr, int batch_size,
  int num_channels, int height, int width)
{
  chkCUDA(cudaMalloc((void **)pptr,
    sizeof(float) * batch_size * num_channels * height * width));
  chkCUDA(cudaMemset(*pptr, 0,
    sizeof(float) * batch_size * num_channels * height * width));
}

static inline void MALLOC_TENSOR_FLOATZ(float **pptr, int batch_size,
  int num_channels, int height, int width)
{
  MALLOC_TENSOR_FLOAT(pptr, batch_size, num_channels, height, width);
  chkCUDA(cudaMemset(*pptr, 0,
    sizeof(float) * batch_size * num_channels * height * width));
}

static inline int CALC_SIZE(int input, int filter, int pad, int stride)
{
  if ((input + pad * 2 - filter) % stride != 0) {
    fprintf(stderr, "%d %d %d %d\n", input, filter, pad, stride);
  }
  assert((input + pad * 2 - filter) % stride == 0);
  return (input + pad * 2 - filter) / stride + 1;
}

static inline void INITIALIZE_TENSOR_URAND(float *ptr, float start, float end, size_t len)
{
  float *tmp = (float *)malloc(sizeof(float) * len);

  for (int i = 0; i < len; i++) {
    tmp[i] = (end - start) * (float)rand() / RAND_MAX  + start;
  }
  cudaMemcpy(ptr, tmp, sizeof(float) * len, cudaMemcpyHostToDevice);
  free(tmp);
}

static inline void PRINT_TENSOR(float *ptr, size_t len, const char *msg, int nn)
{
  float *tmp = (float *)malloc(sizeof(float) * len);
  cudaMemcpy(tmp, ptr, sizeof(float) * len, cudaMemcpyDeviceToHost);

  fprintf(stderr, "%s\n", msg);
  for (int i = 0; i < len; i++) {
    fprintf(stderr, "%.8f ", tmp[i]);
    if ((i + 1) % nn == 0) {
      fprintf(stderr, "\n");
    }
  }
  fprintf(stderr, "\n");

  free(tmp);
}

static void PRINT_TENSOR4D(float *ptr, int N, int C, int H, int W, const char *msg)
{
  fprintf(stderr, "print tensor4d (%d,%d,%d,%d)\n", N, C, H, W);
  PRINT_TENSOR(ptr, N * C * H * W, msg, 5);
}

static inline int parse_int4_bin(FILE *fp)
{
  char bytes[4];
  bytes[0] = fgetc(fp);
  bytes[1] = fgetc(fp);
  bytes[2] = fgetc(fp);
  bytes[3] = fgetc(fp);

  return *((int *)bytes);
}

static inline float parse_float4_bin(FILE *fp)
{
  char bytes[4];
  bytes[0] = fgetc(fp);
  bytes[1] = fgetc(fp);
  bytes[2] = fgetc(fp);
  bytes[3] = fgetc(fp);

  return *((float *)bytes);
}

static void parse_tensor4d(float **ptr, const char *name)
{
  FILE *of = fopen(name, "rb");
  int shape[4];
  size_t total;

  assert(of != NULL);

  shape[0] = parse_int4_bin(of);
  shape[1] = parse_int4_bin(of);
  shape[2] = parse_int4_bin(of);
  shape[3] = parse_int4_bin(of);

  total = shape[0] * shape[1] * shape[2] * shape[3];

  fprintf(stderr, "reading %s (%d,%d,%d,%d)\n",
    name, shape[0], shape[1], shape[2], shape[3]);

  if (*ptr == NULL) {
    *ptr = (float *)malloc(sizeof(float) * total);
  }

  fread(*ptr, sizeof(float) * total, 1, of);
  /*
  for (int i = 0; i < total; i++) {
    (*ptr)[i] = parse_float4_bin(of);
  }
  */

  fclose(of);
}

static void parse_tensor3d(float **ptr, const char *name)
{
  FILE *of = fopen(name, "rb");
  int shape[3];
  size_t total;

  assert(of != NULL);

  shape[0] = parse_int4_bin(of);
  shape[1] = parse_int4_bin(of);
  shape[2] = parse_int4_bin(of);

  total = shape[0] * shape[1] * shape[2];

  fprintf(stderr, "reading %s (%d,%d,%d,)\n",
    name, shape[0], shape[1], shape[2]);

  if (*ptr == NULL) {
    *ptr = (float *)malloc(sizeof(float) * total);
  }

  fread(*ptr, sizeof(float) * total, 1, of);
  /*
  for (int i = 0; i < total; i++) {
    (*ptr)[i] = parse_float4_bin(of);
  }
  */

  fclose(of);
}


static void parse_tensor2d(float **ptr, const char *name)
{
  FILE *of = fopen(name, "rb");
  int shape[2];
  size_t total;

  assert(of != NULL);

  shape[0] = parse_int4_bin(of);
  shape[1] = parse_int4_bin(of);

  fprintf(stderr, "reading %s (%d,%d,)\n",
    name, shape[0], shape[1]);

  total = shape[0] * shape[1];
  if (*ptr == NULL) {
    *ptr = (float *)malloc(sizeof(float) * total);
  }

  for (int i = 0; i < total; i++) {
    (*ptr)[i] = parse_float4_bin(of);
  }

  fclose(of);
}

static void parse_tensor1d(float **ptr, const char *name)
{
  FILE *of = fopen(name, "rb");
  int shape[1];
  size_t total;

  assert(of != NULL);

  shape[0] = parse_int4_bin(of);

  fprintf(stderr, "reading %s (%d,)\n", name, shape[0]);

  total = shape[0];
  if (*ptr == NULL) {
    *ptr = (float *)malloc(sizeof(float) * total);
  }

  for (int i = 0; i < total; i++) {
    (*ptr)[i] = parse_float4_bin(of);
    // TODO
  }

  fclose(of);
}

static void WRITE_TENSORND(float *d_ptr, int n, int shape[], const char *name)
{
  FILE *of = fopen(name, "wb");
  size_t total = 1;
  float *ptr;
  int i;

  assert(of != NULL);

  fprintf(stderr, "writing %s (%d, %d, %d,)\n",
    name, shape[0], shape[1], shape[2]);

  fwrite(shape, sizeof(int) * n, 1, of);

  for (i = 0; i < n; i++) total *= shape[i];
  ptr = (float *)malloc(sizeof(float) * total);
  cudaMemcpy(ptr, d_ptr, sizeof(float) * total, cudaMemcpyDeviceToHost);

  fwrite(ptr, sizeof(float) * total, 1, of);

  free(ptr);
  fclose(of);
}

void **get_global_workspace(size_t bytes);

#endif

