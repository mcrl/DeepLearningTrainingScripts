#ifndef _UTILS_H_
#define _UTILS_H_

#include <stdio.h>
#include <stdlib.h>
#include <assert.h>

#include <cuda.h>
#include <cudnn.h>
#include <cublas_v2.h>
#include <time.h>

#define MIN(a, b) (((a) < (b)) ? (a) : (b))
#define M_PI 3.14159265358979323846

#define START_CNN_TIMER(name) \
  static struct timespec st_##name;\
  do {\
    cudaDeviceSynchronize();\
    clock_gettime(CLOCK_MONOTONIC, &st_##name);\
  } while (0)

#define STOP_CNN_TIMER(name) \
  static struct timespec ed_##name;\
  do {\
    cudaDeviceSynchronize();\
    clock_gettime(CLOCK_MONOTONIC, &ed_##name);\
    l->name += diff_timespec_ms(st_##name, ed_##name);\
  } while (0)

#define chkCUDA(exp) \
  do {\
    cudaError_t status = (exp);\
    if (status != cudaSuccess) {\
      fprintf(stderr, "[%s] Error on line %d: %s\n",\
          __FILE__, __LINE__, cudaGetErrorString(status));\
      exit(EXIT_FAILURE);\
    }\
  } while (0)

#define chkCUDNN(exp) \
  do {\
    cudnnStatus_t status = (exp);\
    if (status != CUDNN_STATUS_SUCCESS) {\
      fprintf(stderr, "[%s] Error on line %d: %s\n",\
          __FILE__, __LINE__, cudnnGetErrorString(status));\
      exit(EXIT_FAILURE);\
    }\
  } while (0)

#define chkCUBLAS(exp) \
  do {\
    cublasStatus_t status = (exp);\
    if (status != CUBLAS_STATUS_SUCCESS) {\
      fprintf(stderr, "[%s] Error on line %d: %d\n",\
          __FILE__, __LINE__, (int)status);\
      exit(EXIT_FAILURE);\
    }\
  } while (0)

static inline float diff_timespec_ms(struct timespec st, struct timespec ed)
{
  return (ed.tv_sec - st.tv_sec) * 1000 + (ed.tv_nsec - st.tv_nsec) / 1000000.0;
}

static inline void MALLOC_TENSOR_INT(
    int **pptr, int batch_size, int num_channels, int height, int width)
{
  size_t sz = sizeof(int) * batch_size * num_channels * height * width;

  chkCUDA(cudaMalloc((void **)pptr, sz));
  chkCUDA(cudaMemset(*pptr, 0, sz));
}

static inline void MALLOC_TENSOR_FLOAT(
    float **pptr, int batch_size, int num_channels, int height, int width)
{
  size_t sz = sizeof(float) * batch_size * num_channels * height * width;

  chkCUDA(cudaMalloc((void **)pptr, sz));
  chkCUDA(cudaMemset(*pptr, 0, sz));
}

static inline int CALC_SIZE(int input, int filter, int pad, int stride)
{
  return (input + pad * 2 - filter) / stride + 1;
}

static inline void INITIALIZE_TENSOR_URAND(
    float *ptr, float start, float end, size_t len)
{
  float *tmp = (float *)malloc(sizeof(float) * len);

  for (int i = 0; i < len; i++) {
    tmp[i] = (end - start) * (float)rand() / RAND_MAX + start;
  }

  cudaMemcpy(ptr, tmp, sizeof(float) * len, cudaMemcpyHostToDevice);
  free(tmp);
}

static inline void INITIALIZE_RAND(float *ptr, size_t len)
{
  for (int i = 0; i < len; i++) {
    ptr[i] = ((float)rand() / RAND_MAX - 0.5f);
  }
}

static inline void INITIALIZE_RAND_SCALE(float *ptr, size_t len, float scale)
{
  for (int i = 0; i < len; i++) {
    ptr[i] = (((float)rand() / RAND_MAX) * 2 - 1) * scale;
  }
}

static float gauss(void)
{
  float x = (float)rand() / RAND_MAX;
  float y = (float)rand() / RAND_MAX;
  float z = sqrt(-2 * log(x)) * cos(2 * M_PI * y);
  return z;
}

static inline void INITIALIZE_RAND_NORM_SCALE(float *ptr, size_t len, float scale)
{
  for (int i = 0; i < len; i++) {
    ptr[i] = gauss() * scale; 
  }
}

static inline void INITIALIZE_CONST(float *ptr, size_t len, float cst)
{
  for (int i = 0; i < len; i++) {
    ptr[i] = cst;
  }
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

static void PRINT_TENSOR_WITH_DESC(float *ptr, cudnnTensorDescriptor_t desc, const char *msg)
{
  cudnnDataType_t type;
  int n, c, h, w, nt, nc, nh, nw;

  chkCUDNN(cudnnGetTensor4dDescriptor(desc, &type, &n, &c, &h, &w, &nt, &nc, &nh, &nw));

  fprintf(stderr, "print tensor4d (%d,%d,%d,%d)\n", n, c, h, w);
  PRINT_TENSOR(ptr, n * c * h * w, msg, 5);
}

static void PRINT_FILTER_WITH_DESC(float *ptr, cudnnFilterDescriptor_t desc, const char *msg)
{
  cudnnDataType_t type;
  cudnnTensorFormat_t format;
  int k, c, r, s;

  chkCUDNN(cudnnGetFilter4dDescriptor(desc, &type, &format, &k, &c, &r, &s));

  fprintf(stderr, "print tensor4d (%d,%d,%d,%d)\n", k, c, r, s);
  PRINT_TENSOR(ptr, k * c * r * s, msg, 5);
}

static void PRINT_TENSOR4D(float *ptr, int N, int C, int H, int W, const char *msg)
{
  fprintf(stderr, "print tensor4d (%d,%d,%d,%d)\n", N, C, H, W);
  PRINT_TENSOR(ptr, N * C * H * W, msg, 5);
}

static void WRITE_TENSORND(float *d_ptr, int n, int shape[], const char *name)
{
  FILE *of = fopen(name, "wb");
  size_t total = 1;
  float *ptr;
  int i;

  assert(of);

  fprintf(stderr, "writing %s (%d, %d, %d,)\n", name, shape[0], shape[1], shape[2]);
  fwrite(shape, sizeof(int) * n, 1, of);

  for (i = 0; i < n; i++) {
    total *= shape[i];
  }

  ptr = (float *)malloc(sizeof(float) * total);
  cudaMemcpy(ptr, d_ptr, sizeof(float) * total, cudaMemcpyDeviceToHost);
  fwrite(ptr, sizeof(float) * total, 1, of);

  free(ptr);
  fclose(of);
}

static void PRINT_TENSOR_DESC(cudnnTensorDescriptor_t t)
{
  cudnnDataType_t type;
  int n, c, h, w, nt, nc, nh, nw;

  chkCUDNN(cudnnGetTensor4dDescriptor(t, &type, &n, &c, &h, &w, &nt, &nc, &nh, &nw));

  printf("%d, (%d, %d, %d, %d), (%d, %d, %d, %d)\n", type, n, c, h, w, nt, nc, nh, nw);
}

static int TENSOR_SIZE(cudnnTensorDescriptor_t t)
{
  cudnnDataType_t type;
  int n, c, h, w, nt, nc, nh, nw;

  chkCUDNN(cudnnGetTensor4dDescriptor(t, &type, &n, &c, &h, &w, &nt, &nc, &nh, &nw));

  return n * c * h * w;
}

#endif // _UTILS_H_
