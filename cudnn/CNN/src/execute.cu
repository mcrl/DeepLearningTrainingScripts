#include <stdlib.h>
#include <stdbool.h>
#include <assert.h>

#include <cuda.h>
#include <cudnn.h>
#include <cublas_v2.h>

#include "execute.h"
#include "memory.h"
#include "utils.h"

extern int node_id;
extern int num_nodes;
extern int num_devices;

extern cudaStream_t memory_stream[MAX_NDEV];
cudaStream_t kernel_stream[MAX_NDEV];
cudnnHandle_t cudnn_handle[MAX_NDEV];
cublasHandle_t cublas_handle[MAX_NDEV];

static const float one_float32 = 1.0;
static const float zero_float32 = 0.0;

#define __1 ( (const void *)&one_float32 )
#define __0 ( (const void *)&zero_float32 )

static inline int distribute(int n, int dev)
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
  else printf("initialize stream executer\n");

  chkCUDA(cudaGetDeviceCount(&num_devices));

  num_devices = MIN(num_devices, MAX_NDEV);

  printf("num_devices : %d\n", num_devices);

  for (int dev = 0; dev < num_devices; dev++) {
    chkCUDA(cudaSetDevice(dev));
    chkCUDA(cudaStreamCreate(&kernel_stream[dev]));

    chkCUDNN(cudnnCreate(&cudnn_handle[dev]));
    chkCUDNN(cudnnSetStream(cudnn_handle[dev], kernel_stream[dev]));

    chkCUBLAS(cublasCreate(&cublas_handle[dev]));
    chkCUBLAS(cublasSetStream(cublas_handle[dev], kernel_stream[dev]));
  }

  initialized = true;

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
int execute_act_bwd(
    cudnnActivationDescriptor_t actDesc,
    gpu_mem y, gpu_mem dy, gpu_mem x, gpu_mem dx)
{
  assert(y->obj_type == DATA);
  assert(dy->obj_type == DATA_GRADIENT);
  assert(x->obj_type == DATA);
  assert(dx->obj_type == DATA_GRADIENT);

  for (int dev = 0; dev < num_devices; dev++) {
    chkCUDA(cudaSetDevice(dev));

    chkCUDNN(cudnnActivationBackward(
          cudnn_handle[dev],
          actDesc,
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

int execute_act_fwd(
    cudnnActivationDescriptor_t actDesc,
    gpu_mem x, gpu_mem y)
{
  assert(x->obj_type == DATA);
  assert(y->obj_type == DATA);

  for (int dev = 0; dev < num_devices; dev++) {
    chkCUDA(cudaSetDevice(dev));

    chkCUDNN(cudnnActivationForward(
          cudnn_handle[dev],
          actDesc,
          __1,
          x->tensor_desc[dev],
          x->dev_ptr[dev],
          __0,
          y->tensor_desc[dev],
          y->dev_ptr[dev]));
  }

  return 0;
}

/* Batch Normalization */
int execute_bn_bwd(
    cudnnBatchNormMode_t mode,
    double eps,
    gpu_mem x, gpu_mem dy, gpu_mem dx,
    gpu_mem w, gpu_mem dw, gpu_mem db,
    gpu_mem s_mean, gpu_mem s_var)
{
  assert(x->obj_type == DATA);
  assert(dy->obj_type == DATA_GRADIENT);
  assert(dx->obj_type == DATA_GRADIENT);
  assert(w->obj_type == BN_PARAM);
  assert(dw->obj_type == BN_PARAM_GRADIENT);
  assert(db->obj_type == BN_PARAM_GRADIENT);
  assert(s_mean->obj_type == BN_PARAM);
  assert(s_var->obj_type == BN_PARAM);

  for (int dev = 0; dev < num_devices; dev++) {
    chkCUDA(cudaSetDevice(dev));

    chkCUDNN(cudnnBatchNormalizationBackward(
          cudnn_handle[dev],
          mode,
          __1,
          __0,
          __1,
          __0,
          x->tensor_desc[dev],
          x->dev_ptr[dev],
          dy->tensor_desc[dev],
          dy->dev_ptr[dev],
          dx->tensor_desc[dev],
          dx->dev_ptr[dev],
          w->tensor_desc[dev],
          w->dev_ptr[dev],
          dw->dev_ptr[dev],
          db->dev_ptr[dev],
          eps,
          s_mean->dev_ptr[dev],
          s_var->dev_ptr[dev]));
  }

  return 0;
}

int execute_bn_fwd(
    cudnnBatchNormMode_t mode,
    double eaf, double eps,
    gpu_mem x, gpu_mem y, gpu_mem w, gpu_mem b,
    gpu_mem r_mean, gpu_mem r_var,
    gpu_mem s_mean, gpu_mem s_var)
{
  assert(x->obj_type == DATA);
  assert(y->obj_type == DATA);
  assert(w->obj_type == BN_PARAM);
  assert(b->obj_type == BN_PARAM);
  assert(r_mean->obj_type == BN_PARAM);
  assert(r_var->obj_type == BN_PARAM);
  assert(s_mean->obj_type == BN_PARAM);
  assert(s_var->obj_type == BN_PARAM);

  for (int dev = 0; dev < num_devices; dev++) {
    chkCUDA(cudaSetDevice(dev));

    chkCUDNN(cudnnBatchNormalizationForwardTraining(
          cudnn_handle[dev],
          mode,
          __1,
          __0,
          x->tensor_desc[dev],
          x->dev_ptr[dev],
          y->tensor_desc[dev],
          y->dev_ptr[dev],
          w->tensor_desc[dev],
          w->dev_ptr[dev],
          b->dev_ptr[dev],
          eaf,
          r_mean->dev_ptr[dev],
          r_var->dev_ptr[dev],
          eps,
          s_mean->dev_ptr[dev],
          s_var->dev_ptr[dev]));
  }

  return 0;
}

/* Bias */
int execute_bias_bwd(gpu_mem dy, gpu_mem db)
{
  assert(dy->obj_type == DATA_GRADIENT);
  assert(db->obj_type == WEIGHT_GRADIENT);

  for (int dev = 0; dev < num_devices; dev++) {
    chkCUDA(cudaSetDevice(dev));

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
      chkCUDA(cudaSetDevice(dev));
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
    chkCUDA(cudaSetDevice(dev));

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

/* Branch */
int execute_branch_bwd(
    cudnnOpTensorDescriptor_t opDesc,
    int fan_out, gpu_mem dy[], gpu_mem dx)
{
  assert(fan_out > 1);
  for (int i = 0; i < fan_out; i++) {
    assert(dy[i]->obj_type == DATA_GRADIENT);
  }
  assert(dx->obj_type == DATA_GRADIENT);

  for (int dev = 0; dev < num_devices; dev++) {
    chkCUDA(cudaSetDevice(dev));

    chkCUDNN(cudnnOpTensor(
          cudnn_handle[dev],
          opDesc,
          __1,
          dy[0]->tensor_desc[dev],
          dy[0]->dev_ptr[dev],
          __1,
          dy[1]->tensor_desc[dev],
          dy[1]->dev_ptr[dev],
          __0,
          dx->tensor_desc[dev],
          dx->dev_ptr[dev]));

    for (int i = 2; i < fan_out; i++) {
      chkCUDNN(cudnnAddTensor(
            cudnn_handle[dev],
            __1,
            dy[i]->tensor_desc[dev],
            dy[i]->dev_ptr[dev],
            __1,
            dx->tensor_desc[dev],
            dx->dev_ptr[dev]));
    }
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
    chkCUDA(cudaSetDevice(dev));

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
    chkCUDA(cudaSetDevice(dev));

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
      chkCUDA(cudaSetDevice(dev));

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
    chkCUDA(cudaSetDevice(dev));

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

int execute_get_conv_bwd_data_algo(
    cudnnConvolutionDescriptor_t convDesc,
    gpu_mem w, gpu_mem dy, gpu_mem dx,
    cudnnConvolutionBwdDataAlgo_t *algo)
{
  assert(w->obj_type == WEIGHT);
  assert(dy->obj_type == DATA_GRADIENT);
  assert(dx->obj_type == DATA_GRADIENT);

  chkCUDNN(cudnnGetConvolutionBackwardDataAlgorithm(
        cudnn_handle[0],
        w->filter_desc,
        dy->tensor_desc[0],
        convDesc,
        dx->tensor_desc[0],
        CUDNN_CONVOLUTION_BWD_DATA_PREFER_FASTEST,
        0,
        algo));

  return 0;
}

int execute_get_conv_bwd_data_ws_size(
    cudnnConvolutionDescriptor_t convDesc,
    cudnnConvolutionBwdDataAlgo_t algo,
    gpu_mem w, gpu_mem dy, gpu_mem dx,
    size_t *ws_size)
{
  assert(w->obj_type == WEIGHT);
  assert(dy->obj_type == DATA_GRADIENT);
  assert(dx->obj_type == DATA_GRADIENT);

  chkCUDNN(cudnnGetConvolutionBackwardDataWorkspaceSize(
        cudnn_handle[0],
        w->filter_desc,
        dy->tensor_desc[0],
        convDesc,
        dx->tensor_desc[0],
        algo,
        ws_size));

  return 0;
}

int execute_get_conv_bwd_filter_algo(
    cudnnConvolutionDescriptor_t convDesc,
    gpu_mem x, gpu_mem dy, gpu_mem dw,
    cudnnConvolutionBwdFilterAlgo_t *algo)
{
  assert(x->obj_type == DATA);
  assert(dy->obj_type == DATA_GRADIENT);
  assert(dw->obj_type == WEIGHT_GRADIENT);

  chkCUDNN(cudnnGetConvolutionBackwardFilterAlgorithm(
        cudnn_handle[0],
        x->tensor_desc[0],
        dy->tensor_desc[0],
        convDesc,
        dw->filter_desc,
        CUDNN_CONVOLUTION_BWD_FILTER_PREFER_FASTEST,
        0,
        algo));

  return 0;
}

int execute_get_conv_bwd_filter_ws_size(
    cudnnConvolutionDescriptor_t convDesc,
    cudnnConvolutionBwdFilterAlgo_t algo,
    gpu_mem x, gpu_mem dy, gpu_mem dw,
    size_t *ws_size)
{
  assert(x->obj_type == DATA);
  assert(dy->obj_type == DATA_GRADIENT);
  assert(dw->obj_type == WEIGHT_GRADIENT);

  chkCUDNN(cudnnGetConvolutionBackwardFilterWorkspaceSize(
        cudnn_handle[0],
        x->tensor_desc[0],
        dy->tensor_desc[0],
        convDesc,
        dw->filter_desc,
        algo,
        ws_size));

  return 0;
}

int execute_get_conv_fwd_algo(
    cudnnConvolutionDescriptor_t convDesc,
    gpu_mem x, gpu_mem w, gpu_mem y,
    cudnnConvolutionFwdAlgo_t *algo)
{
  assert(x->obj_type == DATA);
  assert(w->obj_type == WEIGHT);
  assert(y->obj_type == DATA);

  chkCUDNN(cudnnGetConvolutionForwardAlgorithm(
        cudnn_handle[0],
        x->tensor_desc[0],
        w->filter_desc,
        convDesc,
        y->tensor_desc[0],
        CUDNN_CONVOLUTION_FWD_PREFER_FASTEST,
        0,
        algo));

  return 0;
}

int execute_get_conv_fwd_ws_size(
    cudnnConvolutionDescriptor_t convDesc,
    cudnnConvolutionFwdAlgo_t algo,
    gpu_mem x, gpu_mem w, gpu_mem y,
    size_t *ws_size)
{
  assert(x->obj_type == DATA);
  assert(w->obj_type == WEIGHT);
  assert(y->obj_type == DATA);

  chkCUDNN(cudnnGetConvolutionForwardWorkspaceSize(
        cudnn_handle[0],
        x->tensor_desc[0],
        w->filter_desc,
        convDesc,
        y->tensor_desc[0],
        algo,
        ws_size));

  return 0;
}

/* Element-wise Operation */
int execute_elt(
    cudnnOpTensorDescriptor_t opDesc,
    gpu_mem x1, gpu_mem x2, gpu_mem y)
{
  for (int dev = 0; dev < num_devices; dev++) {
    chkCUDA(cudaSetDevice(dev));

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
    chkCUDA(cudaSetDevice(dev));

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
    chkCUDA(cudaSetDevice(dev));

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
    chkCUDA(cudaSetDevice(dev));

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
    chkCUDA(cudaSetDevice(dev));

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
    const float learning_rate, gpu_mem dw, gpu_mem w)
{
  assert(dw->obj_type == WEIGHT_GRADIENT || dw->obj_type == BN_PARAM_GRADIENT);
  assert(w->obj_type == WEIGHT || w->obj_type == BN_PARAM);

  float alpha = -learning_rate;

  for (int dev = 0; dev < num_devices; dev++) {
    int num_elements = w->size_in_bytes[dev] / data_type_size(w);
    chkCUDA(cudaSetDevice(dev));

    chkCUBLAS(cublasSaxpy(
          cublas_handle[dev],
          num_elements,
          &alpha,
          (float *)dw->dev_ptr[dev],
          1,
          (float *)w->dev_ptr[dev],
          1));
  }

  return 0;
}

////////////////////////////////////////////////////////////
// CUDA kernel based API
////////////////////////////////////////////////////////////

__global__ void cuda_concat2(
    int batch_size, int channel1, int channel2, int height, int width,
    float *in1, float *in2, float *out)
{
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int channel_out = channel1 + channel2;

  if (tid >= batch_size * channel_out * height * width) return;

  int w = tid % width;
  int h = (tid / width) % height;
  int c = (tid / height / width) % channel_out;
  int n = (tid / height / width / channel_out) % batch_size;

  if (c < channel1) {
    int in_idx =
      n * channel1 * width * height +
      c * width * height +
      h * width + w;
    out[tid] = in1[in_idx];
  }
  else {
    int in_idx =
      n * channel2 * width * height +
      (c - channel1) * width * height +
      h * width + w;
    out[tid] = in2[in_idx];
  }
}

__global__ void cuda_split2(
    int batch_size, int channel1, int channel2, int height, int width,
    float *out, float *in1, float *in2)
{
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int channel_out = channel1 + channel2;

  if (tid >= batch_size * channel_out * height * width) return;

  int w = tid % width;
  int h = (tid / width) % height;
  int c = (tid / height / width) % channel_out;
  int n = (tid / height / width / channel_out) % batch_size;

  if (c < channel1) {
    int in_idx =
      n * channel1 * width * height +
      c * width * height +
      h * width + w;
    in1[in_idx] = out[tid];
  }
  else {
    int in_idx =
      n * channel2 * width * height +
      (c - channel1) * width * height +
      h * width + w;
    in2[in_idx] = out[tid];
  }
}

__global__ void cuda_set_label(
    int batch_size, int class_cnt, int *label_in, float *label)
{
  int tid_x = blockIdx.x * blockDim.x + threadIdx.x;

  if (tid_x >= batch_size * class_cnt) return;

  int l = label_in[tid_x / class_cnt];
  int me = tid_x % class_cnt;

  float val = (l == me) ? -1 : 0;
  label[tid_x] = val;
}

/* Concatenation */
int execute_concat_bwd(int fan_in, gpu_mem dy, gpu_mem dx[])
{
  assert(fan_in > 1);
  for (int i = 0; i < fan_in; i++) {
    assert(dx[i]->obj_type == DATA_GRADIENT);
  }
  assert(dy->obj_type == DATA_GRADIENT);

  int block_size = 256;

  for (int dev = 0; dev < num_devices; dev++) {
    int batch_size = distribute(dy->dim[0], dev);
    int grid_size = (batch_size * dy->dim[1] * dy->dim[2] * dy->dim[3] + block_size - 1) / block_size;

    chkCUDA(cudaSetDevice(dev));

    cuda_split2<<<grid_size, block_size, 0, kernel_stream[dev]>>>(
        batch_size, dx[0]->dim[1], dx[1]->dim[1], dy->dim[2], dy->dim[3],
        (float *)dy->dev_ptr[dev], (float *)dx[0]->dev_ptr[dev], (float *)dx[1]->dev_ptr[dev]);
  }

  return 0;
}

int execute_concat_fwd(int fan_in, gpu_mem x[], gpu_mem y)
{
  assert(fan_in > 1);
  for (int i = 0; i < fan_in; i++) {
    assert(x[i]->obj_type == DATA_GRADIENT);
  }
  assert(y->obj_type == DATA_GRADIENT);

  int block_size = 256;

  for (int dev = 0; dev < num_devices; dev++) {
    int batch_size = distribute(y->dim[0], dev);
    int grid_size = (batch_size * y->dim[1] * y->dim[2] * y->dim[3] + block_size - 1) / block_size;

    chkCUDA(cudaSetDevice(dev));

    cuda_concat2<<<grid_size, block_size, 0, kernel_stream[dev]>>>(
        batch_size, x[0]->dim[1], x[1]->dim[1], y->dim[2], y->dim[3],
        (float *)x[0]->dev_ptr[dev], (float *)x[1]->dev_ptr[dev], (float *)y->dev_ptr[dev]);
  }

  return 0;
}

/* Softmax */
int execute_set_label(gpu_mem l, gpu_mem dy)
{
  assert(l->obj_type == DATA);
  assert(dy->obj_type == DATA_GRADIENT);

  int block_size = 256;

  for (int dev = 0; dev < num_devices; dev++) {
    int batch_size = distribute(l->dim[0], dev);
    int class_size = l->dim[1];
    int grid_size = (batch_size * class_size + block_size - 1) / block_size;

    chkCUDA(cudaSetDevice(dev));

    cuda_set_label<<<grid_size, block_size, 0, kernel_stream[dev]>>>(
        batch_size, class_size, (int *)l->dev_ptr[dev], (float *)dy->dev_ptr[dev]);
  }

  return 0;
}

////////////////////////////////////////////////////////////
// CUDA runtime based API
////////////////////////////////////////////////////////////

int synch_comp()
{
  for (int dev = 0; dev < num_devices; dev++) {
    chkCUDA(cudaSetDevice(dev));
    chkCUDA(cudaStreamSynchronize(kernel_stream[dev]));
  }
  // MPI_Barrier();

  return 0;
}

int synch_comm()
{
  for (int dev = 0; dev < num_devices; dev++) {
    chkCUDA(cudaSetDevice(dev));
    chkCUDA(cudaStreamSynchronize(kernel_stream[dev]));
  }
  // MPI_Barrier();

  return 0;
}

int synch_device()
{
  for (int dev = 0; dev < num_devices; dev++) {
    chkCUDA(cudaSetDevice(dev));
    chkCUDA(cudaDeviceSynchronize());
  }
  // MPI_Barrier();

  return 0;
}
