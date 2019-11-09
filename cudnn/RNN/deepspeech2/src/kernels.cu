
__global__ void sum_padded_seq_2d(float *in, float *out, int len)
{
  int tid_x = blockIdx.x * blockDim.x + threadIdx.x;
  int tid_y = blockIdx.y * blockDim.y + threadIdx.y;

  out[tid_x * len / 2 + tid_y] = in[tid_x * len + tid_y] +
    in[tid_x * len + (len / 2) + tid_y];
}

__global__ void sum_padded_seq_1d(float *in, float *out, int len, int num_rows)
{
  int tid_x = blockIdx.x * blockDim.x + threadIdx.x;
  int i;

  if (tid_x >= len / 2) return;

  for (i = 0; i < num_rows; i++) {
    out[i * len / 2 + tid_x] = in[i * len + tid_x] +
      in[i * len + (len / 2) + tid_x];
  }
}

__global__ void expand_sum_padded_seq_1d(float *in, float *out, int len, int num_rows)
{
  int tid_x = blockIdx.x * blockDim.x + threadIdx.x;
  int i;

  if (tid_x >= len / 2) return;

  for (i = 0; i < num_rows; i++) {
    out[i * len + tid_x] = in[i * (len / 2) + tid_x];
    out[i * len + (len / 2) + tid_x] =  in[i * (len / 2) + tid_x];
  }
}

__global__ void transpose_3d(float *in, float *out, int N, int H, int W)
{
  int n = blockIdx.x * blockDim.x + threadIdx.x;
  int h = blockIdx.y * blockDim.y + threadIdx.y;
  int w = blockIdx.z * blockDim.z + threadIdx.z;

  if (n >= N || h >= H || w >= W) return;
  out[w * N * H + n * H + h] = in[n * H * W + h * W + w];
}

__global__ void transpose_inverse_3d(float *in, float *out, int N, int H, int W)
{
  int n = blockIdx.x * blockDim.x + threadIdx.x;
  int h = blockIdx.y * blockDim.y + threadIdx.y;
  int w = blockIdx.z * blockDim.z + threadIdx.z;

  if (n >= N || h >= H || w >= W) return;
  out[n * H * W + h * W + w] = in[w * N * H + n * H + h];
}

__global__ void snrm2(float *in, float *out, int len, int elem_per_th)
{
  int tid_x = blockIdx.x * blockDim.x + threadIdx.x;
  int num_ths = blockDim.x * gridDim.x;
  int sp = tid_x;
  int i;

  extern __shared__ float shared_sum[];
  float private_sum = 0;

  if (sp >= len) return;

  for (i = sp; elem_per_th > 0; i += num_ths, elem_per_th--) {
    if (i < len) {
      float val = in[i];
      private_sum += val * val;
    }
  }

  shared_sum[threadIdx.x] = private_sum;
  int off = blockDim.x / 2;

  __syncthreads();

  while (off > 0) {
    if (threadIdx.x < off) {
      shared_sum[threadIdx.x] +=
        shared_sum[threadIdx.x + off];
    }
    off /= 2;
    __syncthreads();
  }

  if (threadIdx.x == 0) {
    out[blockIdx.x] = shared_sum[0];
  }
}

