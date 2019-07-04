#ifndef _UTILS_H_
#define _UTILS_H_

#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <math.h>
#include <time.h>

#include <cuda.h>
#include <cudnn.h>
#include <cublas_v2.h>

#define M_PI 3.14159265358979323846

#define MAX(a, b) (((a) < (b)) ? (b) : (a))
#define MIN(a, b) (((a) < (b)) ? (a) : (b))

#define chkCUDA(exp) \
  do {\
    cudaError_t status = (exp);\
    if (status != cudaSuccess) {\
      fprintf(stderr, "[%s] Error on line %d: (code=%d) %s\n",\
          __FILE__, __LINE__, (int)status, cudaGetErrorString(status));\
      exit(EXIT_FAILURE);\
    }\
  } while (0)

#define chkCUDNN(exp) \
  do {\
    cudnnStatus_t status = (exp);\
    if (status != CUDNN_STATUS_SUCCESS) {\
      fprintf(stderr, "[%s] Error on line %d: (code=%d) %s\n",\
          __FILE__, __LINE__, (int)status, cudnnGetErrorString(status));\
      exit(EXIT_FAILURE);\
    }\
  } while (0)

#define chkCUBLAS(exp) \
  do {\
    cublasStatus_t status = (exp);\
    if (status != CUBLAS_STATUS_SUCCESS) {\
      fprintf(stderr, "[%s] Error on line %d: (code=%d) %s\n",\
          __FILE__, __LINE__, (int)status, cublasGetErrorString(status));\
      exit(EXIT_FAILURE);\
    }\
  } while (0)

static inline int exists(const char *fname)
{
  FILE *file;
  if ((file = fopen(fname, "r"))) {
    fclose(file);
    return 1;
  }
  return 0;
}

static inline void verify(float *res, float *ans, int cnt)
{
  const float EPS = 1e-6;
  for (int i = 0; i < cnt; i++) {
    float abs_diff = fabs(res[i]);
    float rel_diff = fabs((res[i] - ans[i])/res[i]);

    if (abs_diff >= EPS && rel_diff >= EPS) {
      printf("%e %e relative_diff = %e\n", res[i], ans[i], rel_diff);
    }

    if (isnan(res[i]) || abs_diff >= EPS && rel_diff >= EPS) {
      fprintf(stderr, "Verification failed at %d, res = %lf, ans = %lf (rel diff = %lf)\n",
          i, res[i], ans[i], rel_diff);
      return;
    }
  }
  fprintf(stderr, "Verification success\n");
}

static inline float diff_timespec_ms(struct timespec st, struct timespec ed)
{
  return (ed.tv_sec - st.tv_sec) * 1000 + (ed.tv_nsec - st.tv_nsec) / 1000000.0;
}

static inline void INITIALIZE_CONST(float *ptr, size_t len, float cst)
{
  for (int i = 0; i < len; i++) {
    ptr[i] = cst;
  }
}

static inline void INITIALIZE_RAND(float *ptr, size_t len)
{
  for (int i = 0; i < len; i++) {
    ptr[i] = (float)rand() / RAND_MAX - 0.5f;
  }
}

static inline void INITIALIZE_RAND_SCALE(float *ptr, size_t len, float scale)
{
  for (int i = 0; i < len; i++) {
    ptr[i] = (((float)rand() / RAND_MAX) * 2 - 1) * scale;
  }
}

static float gauss()
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

static const char *cublasGetErrorString(cublasStatus_t status)
{
  switch (status) {
    case CUBLAS_STATUS_SUCCESS: return "CUBLAS_STATUS_SUCCESS";
    case CUBLAS_STATUS_NOT_INITIALIZED: return "CUBLAS_STATUS_NOT_INITIALIZED";
    case CUBLAS_STATUS_ALLOC_FAILED: return "CUBLAS_STATUS_ALLOC_FAILED";
    case CUBLAS_STATUS_INVALID_VALUE: return "CUBLAS_STATUS_INVALID_VALUE";
    case CUBLAS_STATUS_ARCH_MISMATCH: return "CUBLAS_STATUS_ARCH_MISMATCH";
    case CUBLAS_STATUS_MAPPING_ERROR: return "CUBLAS_STATUS_MAPPING_ERROR";
    case CUBLAS_STATUS_EXECUTION_FAILED: return "CUBLAS_STATUS_EXECUTION_FAILED";
    case CUBLAS_STATUS_INTERNAL_ERROR: return "CUBLAS_STATUS_INTERNAL_ERROR";
    case CUBLAS_STATUS_NOT_SUPPORTED: return "CUBLAS_STATUS_NOT_SUPPORTED";
    case CUBLAS_STATUS_LICENSE_ERROR: return "CUBLAS_STATUS_LICENSE_ERROR";
    default: return "";
  }
}

#endif // _UTILS_H_
