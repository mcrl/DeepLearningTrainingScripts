#include <stdlib.h>
#include <stdbool.h>
#include <assert.h>

#include <cuda.h>
#include <cudnn.h>
#include <cublas_v2.h>

#include "execute.h"
#include "memory.h"
#include "utils.h"

//extern int node_id;
//extern int num_nodes;
//extern int num_workers;
extern int num_devices;

static cudaStream_t kernel_stream[MAX_NDEV];
static cudnnHandle_t cudnn_handle[MAX_NDEV];
static cublasHandle_t cublas_handle[MAX_NDEV];

static float one_float32 = 1.0;
static float zero_float32 = 0.0;

#define __1 ( (const void *)&one_float32 )
#define __0 ( (const void *)&zero_float32 )

static size_t size_of_cudnn_data_type[] = { 4, 8, 2, 1, 4, 4, 1, 4, 32 };

static inline int distribute_batch(int n, int dev)
{
  return (n / num_devices) + (dev < n % num_devices);
}

////////////////////////////////////////////////////////////
// Executer Management API
////////////////////////////////////////////////////////////

int __init_stream_executer()
{
  static bool initialized = false;

  if (initialized) return -1;

  initialized = true;

  for (int dev = 0; dev < num_devices; dev++) {
    chkCUDA(cudaSetDevice(dev));
    chkCUDA(cudaStreamCreate(&kernel_stream[dev]));

    chkCUDNN(cudnnCreate(&cudnn_handle[dev]));
    chkCUDNN(cudnnSetStream(cudnn_handle[dev], kernel_stream[dev]));

    chkCUBLAS(cublasCreate(&cublas_handle[dev]));
    chkCUDNN(cublasSetStream(cublas_handle[dev], kernel_stream[dev]));
  }

  return 0;
}

int __finalize_stream_executer()
{
  for (int dev = 0; dev < num_devices; dev++) {
    chkCUDA(cudaSetDevice(dev));
    chkCUBLAS(cublasDestroy(cublas_handle[dev]));
    chkCUDNN(cudnnDestroy(cudnn_handle[dev]));
    chkCUDA(cudaStreamDestroy(kernel_stream[dev]));
  }

  return 0;
}

////////////////////////////////////////////////////////////
// cuDNN based API
////////////////////////////////////////////////////////////

/* Activation */
int execute_actv_bwd(
    cudnnActivationDescriptor_t actvDesc,
    gpu_mem y, gpu_mem dy, gpu_mem x, gpu_mem dx)
{
  assert(y->obj_type == DATA);
  assert(dy->obj_type == DATA_GRADIENT);
  assert(x->obj_type == DATA);
  assert(dx->obj_type == DATA_GRADIENT);

  for (int dev = 0; dev < num_devices; dev++) {
    chkCUDNN(cudnnActivationBackward(
          cudnn_handle[dev],
          actvDesc,
          __1,
          y->tensor_desc[dev],
          y->dev_ptr[dev],
          dy->tensor_desc[dev],
          dy->dev_ptr[dev],
          x->tensor_desc[dev],
          x->dev_ptr[dev],
          __0,
          dx->tensor_desc[dev],
          dx->dev_ptr[dev]));
  }

  return 0;
}

int execute_actv_fwd(
    cudnnActivationDescriptor_t actvDesc,
    gpu_mem x, gpu_mem y)
{
  assert(x->obj_type == DATA);
  assert(y->obj_type == DATA);

  for (int dev = 0; dev < num_devices; dev++) {
    chkCUDNN(cudnnActivationForward(
          cudnn_handle[dev],
          actvDesc,
          __1,
          x->tensor_desc[dev],
          x->dev_ptr[dev],
          __0,
          y->tensor_desc[dev],
          y->dev_ptr[dev]));
  }

  return 0;
}

/* Bias */
int execute_bias_bwd(gpu_mem dy, gpu_mem db)
{
  assert(dy->obj_type == DATA_GRADIENT);
  assert(db->obj_type == WEIGHT_GRADIENT);

  for (int dev = 0; dev < num_devices; dev++) {
    chkCUDNN(cudnnConvolutionBackwardBias(
          cudnn_handle[dev],
          __1,
          dy->tensor_desc[dev],
          dy->dev_ptr[dev],
          __0,
          db->tensor_desc[dev],
          db->dev_ptr[dev]));
  }

  if (1 * num_devices > 1) {
    for (int dev = 0; dev < num_devices; dev++) {
      chkCUDA(cudaStreamSynchronize(kernel_stream[dev]));
    }

    all_reduce_buffer(db, false);
  }

  return 0;
}

int execute_bias_fwd(gpu_mem b, gpu_mem y)
{
  assert(b->obj_type == WEIGHT);
  assert(y->obj_type == DATA);

  for (int dev = 0; dev < num_devices; dev++) {
    chkCUDNN(cudnnAddTensor(
          cudnn_handle[dev],
          __1,
          b->tensor_desc[dev],
          b->dev_ptr[dev],
          __1,
          y->tensor_desc[dev],
          y->dev_ptr[dev]));
  }

  return 0;
}

/* Convolution */
int execute_conv_bwd_data(
    cudnnConvolutionDescriptor_t convDesc,
    cudnnConvolutionBwdDataAlgo_t algo,
    gpu_mem w, gpu_mem dy,
    gpu_mem dx, gpu_mem workSpace)
{
  assert(w->obj_type == WEIGHT);
  assert(dy->obj_type == DATA_GRADIENT);
  assert(dx->obj_type == DATA_GRADIENT);
  assert(workSpace->obj_type == WORK_SPACE);

  for (int dev = 0; dev < num_devices; dev++) {
    chkCUDNN(cudnnConvolutionBackwardData(
          cudnn_handle[dev],
          __1,
          w->filter_desc,
          w->dev_ptr[dev],
          dy->tensor_desc[dev],
          dy->dev_ptr[dev],
          convDesc,
          algo,
          workSpace->dev_ptr[dev],
          workSpace->size_in_bytes[dev],
          __0,
          dx->tensor_desc[dev],
          dx->dev_ptr[dev]));
  }

  return 0;
}

int execute_conv_bwd_filter(
    cudnnConvolutionDescriptor_t convDesc,
    cudnnConvolutionBwdFilterAlgo_t algo,
    gpu_mem x, gpu_mem dy,
    gpu_mem dw, gpu_mem workSpace)
{
  assert(x->obj_type == DATA);
  assert(dy->obj_type == DATA_GRADIENT);
  assert(dw->obj_type == WEIGHT_GRADIENT);
  assert(workSpace->obj_type == WORK_SPACE);

  for (int dev = 0; dev < num_devices; dev++) {
    chkCUDNN(cudnnConvolutionBackwardFilter(
          cudnn_handle[dev],
          __1,
          x->tensor_desc[dev],
          x->dev_ptr[dev],
          dy->tensor_desc[dev],
          dy->dev_ptr[dev],
          convDesc,
          algo,
          workSpace->dev_ptr[dev],
          workSpace->size_in_bytes[dev],
          __0,
          dw->filter_desc,
          dw->dev_ptr[dev]));
  }

  if (1 * num_devices > 1) {
    for (int dev = 0; dev < num_devices; dev++) {
      chkCUDA(cudaStreamSynchronize(kernel_stream[dev]));
    }

    all_reduce_buffer(dw, false);
  }

  return 0;
}

int execute_conv_fwd(
    cudnnConvolutionDescriptor_t convDesc,
    cudnnConvolutionFwdAlgo_t algo,
    gpu_mem x, gpu_mem w,
    gpu_mem y, gpu_mem workSpace)
{
  assert(x->obj_type == DATA);
  assert(w->obj_type == WEIGHT);
  assert(y->obj_type == DATA);
  assert(workSpace->obj_type == WORK_SPACE);

  for (int dev = 0; dev < num_devices; dev++) {
    chkCUDNN(cudnnConvolutionForward(
          cudnn_handle[dev],
          __1,
          x->tensor_desc[dev],
          x->dev_ptr[dev],
          w->filter_desc,
          w->dev_ptr[dev],
          convDesc,
          algo,
          workSpace->dev_ptr[dev],
          workSpace->size_in_bytes[dev],
          __0,
          y->tensor_desc[dev],
          y->dev_ptr[dev]));
  }

  return 0;
}

/* Element-wise Operation */
int execute_elt_bwd(
    cudnnOpTensorDescriptor_t opDesc,
    gpu_mem y1, gpu_mem y2, gpu_mem x)
{
  assert(x->obj_type == DATA || x->obj_type == DATA_GRADIENT);
  assert(y1->obj_type == DATA || y1->obj_type == DATA_GRADIENT);
  assert(y2->obj_type == DATA || y2->obj_type == DATA_GRADIENT);

  for (int dev = 0; dev < num_devices; dev++) {
    chkCUDNN(cudnnOpTensor(
          cudnn_handle[dev],
          opDesc,
          __1,
          y1->tensor_desc[dev],
          y1->dev_ptr[dev],
          __1,
          y2->tensor_desc[dev],
          y2->dev_ptr[dev],
          __0,
          x->tensor_desc[dev],
          x->dev_ptr[dev]));
  }

  return 0;
}

int execute_elt_fwd(
    cudnnOpTensorDescriptor_t opDesc,
    gpu_mem x1, gpu_mem x2, gpu_mem y)
{
  assert(x1->obj_type == DATA || x1->obj_type == DATA_GRADIENT);
  assert(x2->obj_type == DATA || x2->obj_type == DATA_GRADIENT);
  assert(y->obj_type == DATA || y->obj_type == DATA_GRADIENT);

  for (int dev = 0; dev < num_devices; dev++) {
    chkCUDNN(cudnnOpTensor(
          cudnn_handle[dev],
          opDesc,
          __1,
          x1->tensor_desc[dev],
          x1->dev_ptr[dev],
          __1,
          x2->tensor_desc[dev],
          x2->dev_ptr[dev],
          __0,
          y->tensor_desc[dev],
          y->dev_ptr[dev]));
  }

  return 0;
}

/* Pooling */
int execute_pool_bwd(
    cudnnPoolingDescriptor_t poolDesc,
    gpu_mem y, gpu_mem dy, gpu_mem x, gpu_mem dx)
{
  assert(y->obj_type == DATA);
  assert(dy->obj_type == DATA_GRADIENT);
  assert(x->obj_type == DATA);
  assert(dx->obj_type == DATA_GRADIENT);

  for (int dev = 0; dev < num_devices; dev++) {
    chkCUDNN(cudnnPoolingBackward(
          cudnn_handle[dev],
          poolDesc,
          __1,
          y->tensor_desc[dev],
          y->dev_ptr[dev],
          dy->tensor_desc[dev],
          dy->dev_ptr[dev],
          x->tensor_desc[dev],
          x->dev_ptr[dev],
          __0,
          dx->tensor_desc[dev],
          dx->dev_ptr[dev]));
  }

  return 0;
}

int execute_pool_fwd(
    cudnnPoolingDescriptor_t poolDesc,
    gpu_mem x, gpu_mem y)
{
  assert(x->obj_type == DATA);
  assert(y->obj_type == DATA);

  for (int dev = 0; dev < num_devices; dev++) {
    chkCUDNN(cudnnPoolingForward(
          cudnn_handle[dev],
          poolDesc,
          __1,
          x->tensor_desc[dev],
          x->dev_ptr[dev],
          __0,
          y->tensor_desc[dev],
          y->dev_ptr[dev]));
  }

  return 0;
}

/* Softmax */
int execute_softmax_bwd(
    cudnnSoftmaxAlgorithm_t algo,
    cudnnSoftmaxMode_t mode,
    gpu_mem y, gpu_mem dy, gpu_mem dx)
{
  assert(y->obj_type == DATA);
  assert(dy->obj_type == DATA_GRADIENT);
  assert(dx->obj_type == DATA_GRADIENT);

  for (int dev = 0; dev < num_devices; dev++) {
    chkCUDNN(cudnnSoftmaxBackward(
          cudnn_handle[dev],
          algo,
          mode,
          __1,
          y->tensor_desc[dev],
          y->dev_ptr[dev],
          dy->tensor_desc[dev],
          dy->dev_ptr[dev],
          __0,
          dx->tensor_desc[dev],
          dx->dev_ptr[dev]));
  }

  return 0;
}

int execute_softmax_fwd(
    cudnnSoftmaxAlgorithm_t algo,
    cudnnSoftmaxMode_t mode,
    gpu_mem x, gpu_mem y)
{
  assert(x->obj_type == DATA);
  assert(y->obj_type == DATA);

  for (int dev = 0; dev < num_devices; dev++) {
    chkCUDNN(cudnnSoftmaxForward(
          cudnn_handle[dev],
          algo,
          mode,
          __1,
          x->tensor_desc[dev],
          x->dev_ptr[dev],
          __0,
          y->tensor_desc[dev],
          y->dev_ptr[dev]));
  }

  return 0;
}

////////////////////////////////////////////////////////////
// cuBLAS based API
////////////////////////////////////////////////////////////

/* Update Weight */
int execute_apply_gradient(
    const float learning_rate,
    gpu_mem dw, gpu_mem w)
{
  assert(dw->obj_type == WEIGHT_GRADIENT);
  assert(w->obj_type == WEIGHT);

  float alpha = -learning_rate;

  for (int dev = 0; dev < num_devices; dev++) {
#if USE_CUBLAS_UPDATE
    chkCUBLAS(cublasSaxpy(
          cublas_handle[dev],
          w->size_in_bytes[dev] / size_of_cudnn_data_type(w->data_type),
          &alpha,
          dw->dev_ptr[dev],
          1,
          w->dev_ptr[dev],
          1));
#else
    chkCUDNN(cudnnAddTensor(
          cudnn_handle[dev],
          &alpha,
          dw->tensor_desc[dev],
          dw->dev_ptr[dev],
          __1,
          w->tensor_desc[dev],
          w->dev_ptr[dev]));
#endif
  }

  return 0;
}
