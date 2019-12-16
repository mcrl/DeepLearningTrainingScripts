#ifndef _CUDNN_UTIL_HPP_
#define _CUDNN_UTIL_HPP_
#include <cudnn.h>
#include <cublas_v2.h>
#include <iostream>
#include <cstdio>

using namespace std;

#define checkCUDNN(expression)								\
{															\
	cudnnStatus_t status = (expression);					\
	if (status != CUDNN_STATUS_SUCCESS) {					\
		cerr << __FILE__  << " Error on line " << __LINE__ << " : "		\
		<< cudnnGetErrorString(status) << endl;				\
		exit(EXIT_FAILURE);									\
	}                                                       \
}

#define checkCUDA(exp)										\
{															\
    cudaError_t status = (exp);								\
    if (status != cudaSuccess) {							\
		cerr << __FILE__  << " Error on line " << __LINE__ << " : "		\
        << cudaGetErrorString(status) << endl;				\
		exit(EXIT_FAILURE);									\
    }                                                       \
}

#define checkCUBLAS(exp)										\
{															\
    cublasStatus_t status = (exp);								\
    if (status != CUBLAS_STATUS_SUCCESS) {							\
		cerr << __FILE__  << " Error on line " << __LINE__ << " : "		\
        << status << endl;				\
		exit(EXIT_FAILURE);									\
    }                                                       \
}

static inline void DUMP_TENSOR(float *ptr, string dumpName, size_t len)
{
	float *tmp = (float *)malloc(sizeof(float) * len);
	cudaMemcpy(tmp, ptr, sizeof(float) * len, cudaMemcpyDeviceToHost);
	FILE *file = fopen(dumpName.c_str(), "w");
	if(file) {
		for(int i = 0; i < len; ++i) {
			fprintf(file, "%4.3f\n", tmp[i]);
		}
		fclose(file);
	}
	free(tmp);
}

static inline void PRINT_TENSOR(float *ptr, size_t len, const char *msg, int nn)
{
  float *tmp = (float *)malloc(sizeof(float) * len);
  cudaMemcpy(tmp, ptr, sizeof(float) * len, cudaMemcpyDeviceToHost);

  fprintf(stderr, "%s\n", msg);
  for (int i = 0; i < len; i++) {
    fprintf(stderr, "%4.3f ", tmp[i]);
    if ((i + 1) % nn == 0) {
      fprintf(stderr, "\n");
    }
  }
  fprintf(stderr, "\n");
  free(tmp);
}

static inline int CALC_SIZE(int input, int filter, int pad, int stride)
{
  return (input + pad * 2 - filter) / stride + 1;
}

static inline void MALLOC_TENSOR_FLOAT(float **pptr, int batch_size,
  int num_channels, int height, int width)
{
  checkCUDA(cudaMalloc((void **)pptr,
    sizeof(float) * batch_size * num_channels * height * width));
  checkCUDA(cudaMemset(*pptr, 0,
    sizeof(float) * batch_size * num_channels * height * width));
}

#endif
