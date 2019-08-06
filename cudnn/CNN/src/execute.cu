#include <stdlib.h>
#include <stdbool.h>
#include <assert.h>

#include <cuda.h>
#include <cudnn.h>
#include <cublas_v2.h>
#include <mpi.h>

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

#define __1 ( (const float *)&one_float32 )
#define __0 ( (const float *)&zero_float32 )

#define distribute(n, dev) ( ((n) / num_devices) + ((dev) < (n) % num_devices) )

#define distribute_len(n, dev) \
  ( ((n) / (num_nodes * num_devices)) + ((dev) < (n) % (num_nodes * num_devices)) )

#define distribute_ofs(n, dev) \
  ( ((n) / (num_nodes * num_devices) * (node_id * num_devices + (dev))) + MIN(node_id * num_devices + (dev), (n) % (num_nodes * num_devices)) )

#define test_input(test_func, mem) \
do {\
  assert(mem);\
  assert(test_func(mem));\
  assert((mem)->allocated);\
} while (0)

#define test_output(test_func, mem) \
do {\
  assert(mem);\
  assert(test_func(mem));\
  if (!(mem)->allocated) {\
    alloc_buffer(mem);\
    assert((mem)->allocated);\
  }\
} while (0)

static const char *conv_bwd_data_algo_msg(cudnnConvolutionBwdDataAlgo_t algo) {
  switch (algo) {
    case CUDNN_CONVOLUTION_BWD_DATA_ALGO_0:
      return "CUDNN_CONVOLUTION_BWD_DATA_ALGO_0";
    case CUDNN_CONVOLUTION_BWD_DATA_ALGO_1:
      return "CUDNN_CONVOLUTION_BWD_DATA_ALGO_1";
    case CUDNN_CONVOLUTION_BWD_DATA_ALGO_FFT:
      return "CUDNN_CONVOLUTION_BWD_DATA_ALGO_FFT";
    case CUDNN_CONVOLUTION_BWD_DATA_ALGO_FFT_TILING:
      return "CUDNN_CONVOLUTION_BWD_DATA_ALGO_FFT_TILING";
    case CUDNN_CONVOLUTION_BWD_DATA_ALGO_WINOGRAD:
      return "CUDNN_CONVOLUTION_BWD_DATA_ALGO_WINOGRAD";
    case CUDNN_CONVOLUTION_BWD_DATA_ALGO_WINOGRAD_NONFUSED:
      return "CUDNN_CONVOLUTION_BWD_DATA_ALGO_WINOGRAD_NONFUSED";
    default:
      return "UNKNOWN_ALGO";
  }
}

static const char *conv_bwd_filter_algo_msg(cudnnConvolutionBwdFilterAlgo_t algo) {
  switch (algo) {
    case CUDNN_CONVOLUTION_BWD_FILTER_ALGO_0:
      return "CUDNN_CONVOLUTION_BWD_FILTER_ALGO_0";
    case CUDNN_CONVOLUTION_BWD_FILTER_ALGO_1:
      return "CUDNN_CONVOLUTION_BWD_FILTER_ALGO_1";
    case CUDNN_CONVOLUTION_BWD_FILTER_ALGO_FFT:
      return "CUDNN_CONVOLUTION_BWD_FILTER_ALGO_FFT";
    case CUDNN_CONVOLUTION_BWD_FILTER_ALGO_3:
      return "CUDNN_CONVOLUTION_BWD_FILTER_ALGO_3";
    case CUDNN_CONVOLUTION_BWD_FILTER_ALGO_WINOGRAD:
      return "CUDNN_CONVOLUTION_BWD_FILTER_ALGO_WINOGRAD";
    case CUDNN_CONVOLUTION_BWD_FILTER_ALGO_WINOGRAD_NONFUSED:
      return "CUDNN_CONVOLUTION_BWD_FILTER_ALGO_WINOGRAD_NONFUSED";
    case CUDNN_CONVOLUTION_BWD_FILTER_ALGO_FFT_TILING:
      return "CUDNN_CONVOLUTION_BWD_FILTER_ALGO_FFT_TILING";
    default:
      return "UNKNOWN_ALGO";
  }
}

static const char *conv_fwd_algo_msg(cudnnConvolutionFwdAlgo_t algo) {
  switch (algo) {
    case CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM:
      return "CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM";
    case CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM:
      return "CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM";
    case CUDNN_CONVOLUTION_FWD_ALGO_GEMM:
      return "CUDNN_CONVOLUTION_FWD_ALGO_GEMM";
    case CUDNN_CONVOLUTION_FWD_ALGO_DIRECT:
      return "CUDNN_CONVOLUTION_FWD_ALGO_DIRECT";
    case CUDNN_CONVOLUTION_FWD_ALGO_FFT:
      return "CUDNN_CONVOLUTION_FWD_ALGO_FFT";
    case CUDNN_CONVOLUTION_FWD_ALGO_FFT_TILING:
      return "CUDNN_CONVOLUTION_FWD_ALGO_FFT_TILING";
    case CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD:
      return "CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD";
    case CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD_NONFUSED:
      return "CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD_NONFUSED";
    default:
      return "UNKNOWN_ALGO";
  }
}

////////////////////////////////////////////////////////////
// Executer Management API
////////////////////////////////////////////////////////////

int __init_stream_executer()
{
  static bool initialized = false;

  if (initialized) return -1;

  chkCUDA(cudaGetDeviceCount(&num_devices));

  num_devices = MIN(num_devices, MAX_NDEV);

  for (int dev = 0; dev < num_devices; dev++) {
    chkCUDA(cudaSetDevice(dev));
    chkCUDA(cudaStreamCreate(&kernel_stream[dev]));

    chkCUDNN(cudnnCreate(&cudnn_handle[dev]));
    chkCUDNN(cudnnSetStream(cudnn_handle[dev], kernel_stream[dev]));

    chkCUBLAS(cublasCreate(&cublas_handle[dev]));
    chkCUBLAS(cublasSetStream(cublas_handle[dev], kernel_stream[dev]));
  }

  initialized = true;

  if (node_id == 0) {
    printf("num_devices : %d\n", num_devices);
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
int execute_act_bwd(
    cudnnActivationDescriptor_t actDesc,
    gpu_mem y, gpu_mem dy, gpu_mem x, gpu_mem dx)
{
  test_input(is_data, y);
  test_input(is_data_grad, dy);
  test_input(is_data, x);
  test_output(is_data_grad, dx);

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
  test_input(is_data, x);
  test_output(is_data, y);

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
  test_input(is_data, x);
  test_input(is_data_grad, dy);
  test_output(is_data_grad, dx);
  test_input(is_bn_param, w);
  test_output(is_bn_param_grad, dw);
  test_output(is_bn_param_grad, db);
  test_output(is_bn_param, s_mean);
  test_output(is_bn_param, s_var);

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
  test_input(is_data, x);
  test_output(is_data, y);
  test_input(is_bn_param, w);
  test_input(is_bn_param, b);
  test_output(is_bn_param, r_mean);
  test_output(is_bn_param, r_var);
  test_output(is_bn_param, s_mean);
  test_output(is_bn_param, s_var);

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
  test_input(is_data_grad, dy);
  test_output(is_weight_grad, db);

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

  if (num_nodes * num_devices == 1) return 0;

  synch_comp();

  return all_reduce_buffer(db, false);
}

int execute_bias_fwd(gpu_mem b, gpu_mem y)
{
  test_input(is_weight, b);
  test_output(is_data, y);

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
    test_input(is_data_grad, dy[i]);
  }
  test_output(is_data_grad, dx);

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
  test_input(is_weight, w);
  test_input(is_data_grad, dy);
  test_output(is_data_grad, dx);
  test_input(is_work_space, workSpace);

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
  test_input(is_data, x);
  test_input(is_data_grad, dy);
  test_output(is_weight_grad, dw);
  test_input(is_work_space, workSpace);

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

  if (num_nodes * num_devices == 1) return 0;

  synch_comp();

  return all_reduce_buffer(dw, false);
}

int execute_conv_fwd(
    cudnnConvolutionDescriptor_t convDesc,
    cudnnConvolutionFwdAlgo_t algo,
    gpu_mem x, gpu_mem w,
    gpu_mem y, gpu_mem workSpace)
{
  test_input(is_data, x);
  test_input(is_weight, w);
  test_output(is_data, y);
  test_input(is_work_space, workSpace);

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
#if 0
  chkCUDNN(cudnnGetConvolutionBackwardDataAlgorithm(
        cudnn_handle[0],
        w->filter_desc,
        dy->tensor_desc[0],
        convDesc,
        dx->tensor_desc[0],
        CUDNN_CONVOLUTION_BWD_DATA_PREFER_FASTEST,
        0,
        algo));

  if (*algo == CUDNN_CONVOLUTION_BWD_DATA_ALGO_FFT_TILING) {
    int height = w->dim[2];
    int width = w->dim[3];

    if (height == 3 && width == 3 || height == 5 && width == 5) {
      *algo = CUDNN_CONVOLUTION_BWD_DATA_ALGO_WINOGRAD_NONFUSED;
    }
    else {
      *algo = CUDNN_CONVOLUTION_BWD_DATA_ALGO_FFT;
    }
  }
#else
  int count_required;
  chkCUDNN(cudnnGetConvolutionBackwardDataAlgorithmMaxCount(
        cudnn_handle[0], &count_required));

  cudnnConvolutionBwdDataAlgoPerf_t *perf_list =
    (cudnnConvolutionBwdDataAlgoPerf_t *)
    malloc(sizeof(cudnnConvolutionBwdDataAlgoPerf_t) * count_required);

  int count_provided;
  chkCUDNN(cudnnFindConvolutionBackwardDataAlgorithm(
        cudnn_handle[0],
        w->filter_desc,
        dy->tensor_desc[0],
        convDesc,
        dx->tensor_desc[0],
        count_required,
        &count_provided,
        perf_list));

  float min_time = 1e9;
  for (int i = 0; i < count_provided; i++) {
    if (perf_list[i].status == CUDNN_STATUS_SUCCESS) {
      if (perf_list[i].time < min_time) {
        min_time = perf_list[i].time;
        *algo = perf_list[i].algo;
      }
    }
  }

  printf("%s(): %s\n", __func__, conv_bwd_data_algo_msg(*algo));
#endif

  return 0;
}

int execute_get_conv_bwd_data_ws_size(
    cudnnConvolutionDescriptor_t convDesc,
    cudnnConvolutionBwdDataAlgo_t algo,
    gpu_mem w, gpu_mem dy, gpu_mem dx,
    size_t *ws_size)
{
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
#if 0
  chkCUDNN(cudnnGetConvolutionBackwardFilterAlgorithm(
        cudnn_handle[0],
        x->tensor_desc[0],
        dy->tensor_desc[0],
        convDesc,
        dw->filter_desc,
        CUDNN_CONVOLUTION_BWD_FILTER_PREFER_FASTEST,
        0,
        algo));

  if (*algo == CUDNN_CONVOLUTION_BWD_FILTER_ALGO_FFT) {
    int height = dw->dim[2];
    int width = dw->dim[3];

    if (height == 3 && width == 3 || height == 5 && width == 5) {
      *algo = CUDNN_CONVOLUTION_BWD_FILTER_ALGO_WINOGRAD_NONFUSED;
    }
    else {
      *algo = CUDNN_CONVOLUTION_BWD_FILTER_ALGO_3;
    }
  }
#else
  int count_required;
  chkCUDNN(cudnnGetConvolutionBackwardFilterAlgorithmMaxCount(
        cudnn_handle[0], &count_required));

  cudnnConvolutionBwdFilterAlgoPerf_t *perf_list =
    (cudnnConvolutionBwdFilterAlgoPerf_t *)
    malloc(sizeof(cudnnConvolutionBwdFilterAlgoPerf_t) * count_required);

  int count_provided;
  chkCUDNN(cudnnFindConvolutionBackwardFilterAlgorithm(
        cudnn_handle[0],
        x->tensor_desc[0],
        dy->tensor_desc[0],
        convDesc,
        dw->filter_desc,
        count_required,
        &count_provided,
        perf_list));

  float min_time = 1e9;
  for (int i = 0; i < count_provided; i++) {
    if (perf_list[i].status == CUDNN_STATUS_SUCCESS) {
      if (perf_list[i].time < min_time) {
        min_time = perf_list[i].time;
        *algo = perf_list[i].algo;
      }
    }
  }

  printf("%s(): %s\n", __func__, conv_bwd_filter_algo_msg(*algo));
#endif

  return 0;
}

int execute_get_conv_bwd_filter_ws_size(
    cudnnConvolutionDescriptor_t convDesc,
    cudnnConvolutionBwdFilterAlgo_t algo,
    gpu_mem x, gpu_mem dy, gpu_mem dw,
    size_t *ws_size)
{
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
#if 0
  chkCUDNN(cudnnGetConvolutionForwardAlgorithm(
        cudnn_handle[0],
        x->tensor_desc[0],
        w->filter_desc,
        convDesc,
        y->tensor_desc[0],
        CUDNN_CONVOLUTION_FWD_PREFER_FASTEST,
        0,
        algo));

  if (*algo == CUDNN_CONVOLUTION_FWD_ALGO_FFT_TILING) {
    int height = w->dim[2];
    int width = w->dim[3];

    if (height == 3 && width == 3 || height == 5 && width == 5) {
      *algo = CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD_NONFUSED;
    }
    else {
      *algo = CUDNN_CONVOLUTION_FWD_ALGO_FFT;
    }
  }
#else
  int count_required;
  chkCUDNN(cudnnGetConvolutionForwardAlgorithmMaxCount(
        cudnn_handle[0], &count_required));

  cudnnConvolutionFwdAlgoPerf_t *perf_list =
    (cudnnConvolutionFwdAlgoPerf_t *)
    malloc(sizeof(cudnnConvolutionFwdAlgoPerf_t) * count_required);

  int count_provided;
  chkCUDNN(cudnnFindConvolutionForwardAlgorithm(
        cudnn_handle[0],
        x->tensor_desc[0],
        w->filter_desc,
        convDesc,
        y->tensor_desc[0],
        count_required,
        &count_provided,
        perf_list));

  float min_time = 1e9;
  for (int i = 0; i < count_provided; i++) {
    if (perf_list[i].status == CUDNN_STATUS_SUCCESS) {
      if (perf_list[i].time < min_time) {
        min_time = perf_list[i].time;
        *algo = perf_list[i].algo;
      }
    }
  }

  printf("%s(): %s\n", __func__, conv_fwd_algo_msg(*algo));
#endif

  return 0;
}

int execute_get_conv_fwd_ws_size(
    cudnnConvolutionDescriptor_t convDesc,
    cudnnConvolutionFwdAlgo_t algo,
    gpu_mem x, gpu_mem w, gpu_mem y,
    size_t *ws_size)
{
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

/* Dropout */
int execute_dropout_bwd(
    cudnnDropoutDescriptor_t drDesc[],
    gpu_mem dy, gpu_mem dx, gpu_mem reserveSpace)
{
  test_input(is_data_grad, dy);
  test_output(is_data_grad, dx);
  test_input(is_reserve_space, reserveSpace);

  for (int dev = 0; dev < num_devices; dev++) {
    chkCUDA(cudaSetDevice(dev));

    chkCUDNN(cudnnDropoutBackward(
          cudnn_handle[dev],
          drDesc[dev],
          dy->tensor_desc[dev],
          dy->dev_ptr[dev],
          dx->tensor_desc[dev],
          dx->dev_ptr[dev],
          reserveSpace->dev_ptr[dev],
          reserveSpace->size_in_bytes[dev]));
  }

  return 0;
}

int execute_dropout_fwd(
    cudnnDropoutDescriptor_t drDesc[],
    gpu_mem x, gpu_mem y, gpu_mem reserveSpace)
{
  test_input(is_data_grad, x);
  test_output(is_data_grad, y);
  test_input(is_reserve_space, reserveSpace);

  for (int dev = 0; dev < num_devices; dev++) {
    chkCUDA(cudaSetDevice(dev));

    chkCUDNN(cudnnDropoutForward(
          cudnn_handle[dev],
          drDesc[dev],
          x->tensor_desc[dev],
          x->dev_ptr[dev],
          y->tensor_desc[dev],
          y->dev_ptr[dev],
          reserveSpace->dev_ptr[dev],
          reserveSpace->size_in_bytes[dev]));
  }

  return 0;
}

int execute_set_dropout(
    cudnnDropoutDescriptor_t drDesc[],
    float rate, unsigned long long seed,
    gpu_mem states)
{
  test_output(is_reserve_space, states);

  for (int dev = 0; dev < num_devices; dev++) {
    chkCUDA(cudaSetDevice(dev));

    chkCUDNN(cudnnSetDropoutDescriptor(
          drDesc[dev],
          cudnn_handle[dev],
          rate,
          states->dev_ptr[dev],
          states->size_in_bytes[dev],
          seed));
  }

  return 0;
}

int execute_get_dropout_st_size(size_t *st_size)
{
  *st_size = 0;

  for (int dev = 0; dev < num_devices; dev++) {
    chkCUDA(cudaSetDevice(dev));

    size_t size_in_bytes;
    chkCUDNN(cudnnDropoutGetStatesSize(
          cudnn_handle[dev], &size_in_bytes));

    *st_size = MAX(*st_size, size_in_bytes);
  }

  return 0;
}

int execute_get_dropout_rs_size(gpu_mem x, size_t *rs_size)
{
  *rs_size = 0;

  for (int dev = 0; dev < num_devices; dev++) {
    chkCUDA(cudaSetDevice(dev));

    size_t size_in_bytes;
    chkCUDNN(cudnnDropoutGetReserveSpaceSize(
          x->tensor_desc[dev], &size_in_bytes));

    *rs_size = MAX(*rs_size, size_in_bytes);
  }

  return 0;
}

/* Element-wise Operation */
int execute_elt(
    cudnnOpTensorDescriptor_t opDesc,
    gpu_mem x1, gpu_mem x2, gpu_mem y)
{
  test_input(is_data_or_data_grad, x1);
  test_input(is_data_or_data_grad, x2);
  test_output(is_data_or_data_grad, y);

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
  test_input(is_data, y);
  test_input(is_data_grad, dy);
  test_input(is_data, x);
  test_output(is_data_grad, dx);

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
  test_input(is_data, x);
  test_output(is_data, y);

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
  test_input(is_data, y);
  test_input(is_data_grad, dy);
  test_input(is_data_grad, dx);

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
  test_input(is_data, x);
  test_output(is_data, y);

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

static void enable_tensor_core()
{
#ifdef USE_TENSOR_CORE
  static bool initialized = false;

  if (!initialized) {
    for (int dev = 0; dev < num_devices; dev++) {
      chkCUDA(cudaSetDevice(dev));
      cublasSetMathMode(cublas_handle[dev], CUBLAS_TENSOR_OP_MATH);
    }
    initialized = true;
  }
#endif
}

/* Linear */
int execute_linear_bwd_data(gpu_mem w, gpu_mem dy, gpu_mem dx)
{
  test_input(is_weight, w);
  test_input(is_data_grad, dy);
  test_output(is_data_grad, dx);

  enable_tensor_core();

  int m = dx->dim[1];
  int n = dx->dim[0];
  int k = dy->dim[1];

  for (int dev = 0; dev < num_devices; dev++) {
    chkCUDA(cudaSetDevice(dev));

    chkCUBLAS(cublasSgemm(
          cublas_handle[dev],
          CUBLAS_OP_N, CUBLAS_OP_N,
          m, distribute_len(n, dev), k,
          __1,
          (float *)w->dev_ptr[dev], m,
          (float *)dy->dev_ptr[dev], k,
          __0,
          (float *)dx->dev_ptr[dev], m));
  }

  return 0;
}

int execute_linear_bwd_weight(gpu_mem x, gpu_mem dy, gpu_mem dw)
{
  test_input(is_data, x);
  test_input(is_data_grad, dy);
  test_output(is_weight_grad, dw);

  enable_tensor_core();

  int m = x->dim[1];
  int n = dy->dim[1];
  int k = x->dim[0];

  for (int dev = 0; dev < num_devices; dev++) {
    chkCUDA(cudaSetDevice(dev));

    chkCUBLAS(cublasSgemm(
          cublas_handle[dev],
          CUBLAS_OP_N, CUBLAS_OP_T,
          m, n, distribute_len(k, dev),
          __1,
          (float *)x->dev_ptr[dev], m,
          (float *)dy->dev_ptr[dev], n,
          __0,
          (float *)dw->dev_ptr[dev], m));
  }

  if (num_nodes * num_devices == 1) return 0;

  synch_comp();

  return all_reduce_buffer(dw, false);
}

int execute_linear_fwd(gpu_mem x, gpu_mem w, gpu_mem y)
{
  test_input(is_data, x);
  test_input(is_weight, w);
  test_output(is_data, y);

  enable_tensor_core();

  int m = y->dim[1];
  int n = x->dim[0];
  int k = x->dim[1];

  for (int dev = 0; dev < num_devices; dev++) {
    chkCUDA(cudaSetDevice(dev));

    chkCUBLAS(cublasSgemm(
          cublas_handle[dev],
          CUBLAS_OP_T, CUBLAS_OP_N,
          m, distribute_len(n, dev), k,
          __1,
          (float *)w->dev_ptr[dev], k,
          (float *)x->dev_ptr[dev], k,
          __0,
          (float *)y->dev_ptr[dev], m));
  }

  return 0;
}

/* Update Weight */
int execute_gradient_descent(
    const float learning_rate, gpu_mem dw, gpu_mem w)
{
  test_input(is_weight_grad_or_param_grad, dw);
  test_output(is_weight_or_param, w);

  enable_tensor_core();

  float alpha = -learning_rate;

  synch_comm();

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
    int batch_size, int height, int width,
    int channel1, int channel2,
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
    int batch_size, int height, int width,
    int channel1, int channel2,
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

__global__ void cuda_concat3(
    int batch_size, int height, int width,
    int channel1, int channel2, int channel3,
    float *in1, float *in2, float *in3, float *out)
{
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int channel_out = channel1 + channel2 + channel3;

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
  else if (c - channel1 < channel2) {
    int in_idx =
      n * channel2 * width * height +
      (c - channel1) * width * height +
      h * width + w;
    out[tid] = in2[in_idx];
  }
  else {
    int in_idx =
      n * channel3 * width * height +
      (c - channel1 - channel2) * width * height +
      h * width + w;
    out[tid] = in3[in_idx];
  }
}

__global__ void cuda_split3(
    int batch_size, int height, int width,
    int channel1, int channel2, int channel3,
    float *out, float *in1, float *in2, float *in3)
{
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int channel_out = channel1 + channel2 + channel3;

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
  else if (c - channel1 < channel2) {
    int in_idx =
      n * channel2 * width * height +
      (c - channel1) * width * height +
      h * width + w;
    in2[in_idx] = out[tid];
  }
  else {
    int in_idx =
      n * channel3 * width * height +
      (c - channel1 - channel2) * width * height +
      h * width + w;
    in3[in_idx] = out[tid];
  }
}

__global__ void cuda_concat4(
    int batch_size, int height, int width,
    int channel1, int channel2, int channel3, int channel4,
    float *in1, float *in2, float *in3, float *in4, float *out)
{
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int channel_out = channel1 + channel2 + channel3 + channel4;

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
  else if (c - channel1 < channel2) {
    int in_idx =
      n * channel2 * width * height +
      (c - channel1) * width * height +
      h * width + w;
    out[tid] = in2[in_idx];
  }
  else if (c - channel1 - channel2 < channel3) {
    int in_idx =
      n * channel3 * width * height +
      (c - channel1 - channel2) * width * height +
      h * width + w;
    out[tid] = in3[in_idx];
  }
  else {
    int in_idx =
      n * channel4 * width * height +
      (c - channel1 - channel2 - channel3) * width * height +
      h * width + w;
    out[tid] = in4[in_idx];
  }
}

__global__ void cuda_split4(
    int batch_size, int height, int width,
    int channel1, int channel2, int channel3, int channel4,
    float *out, float *in1, float *in2, float *in3, float *in4)
{
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int channel_out = channel1 + channel2 + channel3 + channel4;

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
  else if (c - channel1 < channel2) {
    int in_idx =
      n * channel2 * width * height +
      (c - channel1) * width * height +
      h * width + w;
    in2[in_idx] = out[tid];
  }
  else if (c - channel1 - channel2 < channel3) {
    int in_idx =
      n * channel3 * width * height +
      (c - channel1 - channel2) * width * height +
      h * width + w;
    in3[in_idx] = out[tid];
  }
  else {
    int in_idx =
      n * channel4 * width * height +
      (c - channel1 - channel2 - channel3) * width * height +
      h * width + w;
    in4[in_idx] = out[tid];
  }
}

__global__ void cuda_concat6(
    int batch_size, int height, int width,
    int channel1, int channel2, int channel3, int channel4, int channel5, int channel6,
    float *in1, float *in2, float *in3, float *in4, float *in5, float *in6, float *out)
{
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int channel_out = channel1 + channel2 + channel3 + channel4 + channel5 + channel6;

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
  else if (c - channel1 < channel2) {
    int in_idx =
      n * channel2 * width * height +
      (c - channel1) * width * height +
      h * width + w;
    out[tid] = in2[in_idx];
  }
  else if (c - channel1 - channel2 < channel3) {
    int in_idx =
      n * channel3 * width * height +
      (c - channel1 - channel2) * width * height +
      h * width + w;
    out[tid] = in3[in_idx];
  }
  else if (c - channel1 - channel2 - channel3 < channel4) {
    int in_idx =
      n * channel4 * width * height +
      (c - channel1 - channel2 - channel3) * width * height +
      h * width + w;
    out[tid] = in4[in_idx];
  }
  else if (c - channel1 - channel2 - channel3 - channel4 < channel5) {
    int in_idx =
      n * channel5 * width * height +
      (c - channel1 - channel2 - channel3 - channel4) * width * height +
      h * width + w;
    out[tid] = in5[in_idx];
  }
  else if (c - channel1 - channel2 - channel3 - channel4 - channel5 < channel6) {
    int in_idx =
      n * channel6 * width * height +
      (c - channel1 - channel2 - channel3 - channel4 - channel5) * width * height +
      h * width + w;
    out[tid] = in6[in_idx];
  }
}

__global__ void cuda_split6(
    int batch_size, int height, int width,
    int channel1, int channel2, int channel3, int channel4, int channel5, int channel6,
    float *out, float *in1, float *in2, float *in3, float *in4, float *in5, float *in6)
{
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int channel_out = channel1 + channel2 + channel3 + channel4 + channel5 + channel6;

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
  else if (c - channel1 < channel2) {
    int in_idx =
      n * channel2 * width * height +
      (c - channel1) * width * height +
      h * width + w;
    in2[in_idx] = out[tid];
  }
  else if (c - channel1 - channel2 < channel3) {
    int in_idx =
      n * channel3 * width * height +
      (c - channel1 - channel2) * width * height +
      h * width + w;
    in3[in_idx] = out[tid];
  }
  else if (c - channel1 - channel2 - channel3 < channel4) {
    int in_idx =
      n * channel4 * width * height +
      (c - channel1 - channel2 - channel3) * width * height +
      h * width + w;
    in4[in_idx] = out[tid];
  }
  else if (c - channel1 - channel2 - channel3 - channel4 < channel5) {
    int in_idx =
      n * channel5 * width * height +
      (c - channel1 - channel2 - channel3 - channel4) * width * height +
      h * width + w;
    in5[in_idx] = out[tid];
  }
  else if (c - channel1 - channel2 - channel3 - channel4 - channel5 < channel6) {
    int in_idx =
      n * channel6 * width * height +
      (c - channel1 - channel2 - channel3 - channel4 - channel5) * width * height +
      h * width + w;
    in6[in_idx] = out[tid];
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
  test_input(is_data_grad, dy);
  for (int i = 0; i < fan_in; i++) {
    test_output(is_data_grad, dx[i]);
  }

  int block_size = 256;

  for (int dev = 0; dev < num_devices; dev++) {
    int batch_size = distribute_len(dy->dim[0], dev);
    int grid_size = (batch_size * dy->dim[1] * dy->dim[2] * dy->dim[3] + block_size - 1) / block_size;
    chkCUDA(cudaSetDevice(dev));

    if (fan_in == 2) {
      cuda_split2<<<grid_size, block_size, 0, kernel_stream[dev]>>>(
          batch_size, dy->dim[2], dy->dim[3],
          dx[0]->dim[1], dx[1]->dim[1],
          (float *)dy->dev_ptr[dev],
          (float *)dx[0]->dev_ptr[dev],
          (float *)dx[1]->dev_ptr[dev]);
    }
    else if (fan_in == 3) {
      cuda_split3<<<grid_size, block_size, 0, kernel_stream[dev]>>>(
          batch_size, dy->dim[2], dy->dim[3],
          dx[0]->dim[1], dx[1]->dim[1], dx[2]->dim[1],
          (float *)dy->dev_ptr[dev],
          (float *)dx[0]->dev_ptr[dev],
          (float *)dx[1]->dev_ptr[dev],
          (float *)dx[2]->dev_ptr[dev]);
    }
    else if (fan_in == 4) {
      cuda_split4<<<grid_size, block_size, 0, kernel_stream[dev]>>>(
          batch_size, dy->dim[2], dy->dim[3],
          dx[0]->dim[1], dx[1]->dim[1], dx[2]->dim[1], dx[3]->dim[1],
          (float *)dy->dev_ptr[dev],
          (float *)dx[0]->dev_ptr[dev],
          (float *)dx[1]->dev_ptr[dev],
          (float *)dx[2]->dev_ptr[dev],
          (float *)dx[3]->dev_ptr[dev]);
    }
    else if (fan_in == 6) {
      cuda_split6<<<grid_size, block_size, 0, kernel_stream[dev]>>>(
          batch_size, dy->dim[2], dy->dim[3],
          dx[0]->dim[1], dx[1]->dim[1], dx[2]->dim[1], dx[3]->dim[1], dx[4]->dim[1], dx[5]->dim[1],
          (float *)dy->dev_ptr[dev],
          (float *)dx[0]->dev_ptr[dev],
          (float *)dx[1]->dev_ptr[dev],
          (float *)dx[2]->dev_ptr[dev],
          (float *)dx[3]->dev_ptr[dev],
          (float *)dx[4]->dev_ptr[dev],
          (float *)dx[5]->dev_ptr[dev]);
    }
    else {
      assert(0);
    }
  }

  return 0;
}

int execute_concat_fwd(int fan_in, gpu_mem x[], gpu_mem y)
{
  assert(fan_in > 1);
  for (int i = 0; i < fan_in; i++) {
    test_input(is_data, x[i]);
  }
  test_output(is_data, y);

  int block_size = 256;

  for (int dev = 0; dev < num_devices; dev++) {
    int batch_size = distribute_len(y->dim[0], dev);
    int grid_size = (batch_size * y->dim[1] * y->dim[2] * y->dim[3] + block_size - 1) / block_size;
    chkCUDA(cudaSetDevice(dev));

    if (fan_in == 2) {
      cuda_concat2<<<grid_size, block_size, 0, kernel_stream[dev]>>>(
          batch_size, y->dim[2], y->dim[3],
          x[0]->dim[1], x[1]->dim[1],
          (float *)x[0]->dev_ptr[dev],
          (float *)x[1]->dev_ptr[dev],
          (float *)y->dev_ptr[dev]);
    }
    else if (fan_in == 3) {
      cuda_concat3<<<grid_size, block_size, 0, kernel_stream[dev]>>>(
          batch_size, y->dim[2], y->dim[3],
          x[0]->dim[1], x[1]->dim[1], x[2]->dim[1],
          (float *)x[0]->dev_ptr[dev],
          (float *)x[1]->dev_ptr[dev],
          (float *)x[2]->dev_ptr[dev],
          (float *)y->dev_ptr[dev]);
    }
    else if (fan_in == 4) {
      cuda_concat4<<<grid_size, block_size, 0, kernel_stream[dev]>>>(
          batch_size, y->dim[2], y->dim[3],
          x[0]->dim[1], x[1]->dim[1], x[2]->dim[1], x[3]->dim[1],
          (float *)x[0]->dev_ptr[dev],
          (float *)x[1]->dev_ptr[dev],
          (float *)x[2]->dev_ptr[dev],
          (float *)x[3]->dev_ptr[dev],
          (float *)y->dev_ptr[dev]);
    }
    else if (fan_in == 6) {
      cuda_concat6<<<grid_size, block_size, 0, kernel_stream[dev]>>>(
          batch_size, y->dim[2], y->dim[3],
          x[0]->dim[1], x[1]->dim[1], x[2]->dim[1], x[3]->dim[1], x[4]->dim[1], x[5]->dim[1],
          (float *)x[0]->dev_ptr[dev],
          (float *)x[1]->dev_ptr[dev],
          (float *)x[2]->dev_ptr[dev],
          (float *)x[3]->dev_ptr[dev],
          (float *)x[4]->dev_ptr[dev],
          (float *)x[5]->dev_ptr[dev],
          (float *)y->dev_ptr[dev]);
    }
    else {
      assert(0);
    }
  }

  return 0;
}

/* Softmax */
int execute_set_label(gpu_mem l, gpu_mem dy)
{
  test_input(is_data, l);
  test_output(is_data_grad, dy);

  int block_size = 256;

  for (int dev = 0; dev < num_devices; dev++) {
    int batch_size = distribute_len(l->dim[0], dev);
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

  return 0;
}

int synch_comm()
{
  for (int dev = 0; dev < num_devices; dev++) {
    chkCUDA(cudaSetDevice(dev));
    chkCUDA(cudaStreamSynchronize(memory_stream[dev]));
  }

  return 0;
}

int synch_device()
{
  for (int dev = 0; dev < num_devices; dev++) {
    chkCUDA(cudaSetDevice(dev));
    chkCUDA(cudaDeviceSynchronize());
  }
  chkMPI(MPI_Barrier(MPI_COMM_WORLD));

  return 0;
}
