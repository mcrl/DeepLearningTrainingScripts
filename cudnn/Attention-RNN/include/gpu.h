#ifndef __GPU_H__
#define __GPU_H__
#include <cuda_runtime.h>
#include <cudnn.h>
#include <cublas_v2.h>

#include <stdio.h>
#include <stdlib.h>

static const char *_cudaGetErrorEnum(cublasStatus_t error)
{
    switch (error)
    {
        case CUBLAS_STATUS_SUCCESS:
            return "CUBLAS_STATUS_SUCCESS";

        case CUBLAS_STATUS_NOT_INITIALIZED:
            return "CUBLAS_STATUS_NOT_INITIALIZED";

        case CUBLAS_STATUS_ALLOC_FAILED:
            return "CUBLAS_STATUS_ALLOC_FAILED";

        case CUBLAS_STATUS_INVALID_VALUE:
            return "CUBLAS_STATUS_INVALID_VALUE";

        case CUBLAS_STATUS_ARCH_MISMATCH:
            return "CUBLAS_STATUS_ARCH_MISMATCH";

        case CUBLAS_STATUS_MAPPING_ERROR:
            return "CUBLAS_STATUS_MAPPING_ERROR";

        case CUBLAS_STATUS_EXECUTION_FAILED:
            return "CUBLAS_STATUS_EXECUTION_FAILED";

        case CUBLAS_STATUS_INTERNAL_ERROR:
            return "CUBLAS_STATUS_INTERNAL_ERROR";
		default:
			return "<unknown>";
    }

    //return "<unknown>";
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

#define chkCUDA(exp)                                           \
  {                                                            \
    cudaError_t status = (exp);                                \
    if (status != cudaSuccess) {                               \
      fprintf(stderr, "[%s] Error on line %d: %s\n",           \
        __FILE__, __LINE__, cudaGetErrorString(status));       \
      exit(99);                                                \
    }                                                          \
  }

#define chkCUBLAS(exp)                                         \
  {                                                            \
    cublasStatus_t status = (exp);                             \
    if (status != CUBLAS_STATUS_SUCCESS) {                     \
      fprintf(stderr, "[%s] Error on line %d: %s\n",           \
        __FILE__, __LINE__, _cudaGetErrorEnum(status));        \
      exit(99);                                                \
    }                                                          \
  }
#endif //__GPU_H__
