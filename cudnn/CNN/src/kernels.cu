
__global__ void set_label(
    int batch_size, int class_cnt, int *label_in, float *label)
{
  int tid_x = blockIdx.x * blockDim.x + threadIdx.x;

  if (tid_x >= batch_size * class_cnt) return;

  int l = label_in[tid_x / class_cnt];
  int me = tid_x % class_cnt;

  float val = (l == me) ? -1 : 0;
  label[tid_x] = val;
}

__global__ void concat2(
    int batch_size, int channel1, int channel2, int height, int width,
    int fwd, float *in1, float *in2, float *out)
{
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int channel_out = channel1 + channel2;

  if (tid >= batch_size * channel_out * height * width) return;

  int w = tid % width;
  int h = (tid / width) % height;
  int c = (tid / height / width) % channel_out;
  int n = (tid / height / width / channel_out) % batch_size;

  int in_idx;
  float *in;

  if (c < channel1) {
    in_idx = n * channel1 * width * height +
             c * width * height +
             h * width +
             w;
    in = in1;
  }
  else {
    in_idx = n * channel2 * width * height +
             (c - channel1) * width * height +
             h * width +
             w;
    in = in2;
  }

  if (fwd) {
    out[tid] = in[in_idx];
  }
  else {
    in[in_idx] = out[tid];
  }
}

