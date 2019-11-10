#include <stdlib.h>

#include <mpi.h>
#include <nccl.h>
#include <curand.h>
#include "gpu.h"
#include "random.h"
#include "snudnn.h"
#include "utils.h"

cublasHandle_t cublas_handle;
cudnnHandle_t cudnn_handle;

float _one = 1.0f;
float _zero = 0.0f;


cudaStream_t data_stream;
cudaStream_t update_stream;

/*
#define NSTREAM 8
cudaStream_t cuda_streams[NSTREAM];
*/

int rank, nrank;
ncclUniqueId id;
ncclComm_t comm;

static curandGenerator_t curand_gen;

void cuda_scale(float *m, size_t N, float min, float max);



static void mpi_init(int argc, char *argv[])
{
  chkMPI(MPI_Init(&argc, &argv));
  chkMPI(MPI_Comm_rank(MPI_COMM_WORLD, &rank));
  chkMPI(MPI_Comm_size(MPI_COMM_WORLD, &nrank));
}

static void cuda_init()
{
  chkCUDA(cudaSetDevice(rank % 4));
  chkCUDA(cudaStreamCreate(&data_stream));
  chkCUDA(cudaStreamCreate(&update_stream));
}

static void curand_init(int seed)
{
	chkCURAND(curandCreateGenerator(&curand_gen,
				CURAND_RNG_PSEUDO_MTGP32));
	chkCURAND(curandSetPseudoRandomGeneratorSeed(curand_gen, seed));
}

static void cublas_init()
{
  chkCUBLAS(cublasCreate(&cublas_handle));
  chkCUBLAS(cublasSetStream(cublas_handle, data_stream));
}

static void cudnn_init()
{
  chkCUDNN(cudnnCreate(&cudnn_handle));
  chkCUBLAS(cudnnSetStream(cudnn_handle, data_stream));
}

static void nccl_init()
{
  if (rank == 0) chkNCCL(ncclGetUniqueId(&id));
  chkMPI(MPI_Bcast((void *)&id, sizeof(id), MPI_BYTE, 0, MPI_COMM_WORLD));
  chkNCCL(ncclCommInitRank(&comm, nrank, id, rank));
}

void snudnn_init(int argc, char *argv[])
{
  mpi_init(argc, argv);
  cuda_init();
  curand_init(1234);
  cublas_init();
  cudnn_init();
  nccl_init();
}


void snudnn_memseti(int *m, size_t nelem, int val)
{
	int *buffer = malloc(sizeof(int)*nelem);
	for (int i = 0; i < nelem; i++) {
		buffer[i] = val;
	}

	chkCUDA(cudaMemcpyAsync(m, buffer, sizeof(int)*nelem, cudaMemcpyHostToDevice, 0));
	free(buffer);
}

void snudnn_memsetf(float *m, size_t nelem, float val)
{
	float *buffer = malloc(sizeof(float)*nelem);
	for (int i = 0; i < nelem; i++) {
		buffer[i] = val;
	}

	chkCUDA(cudaMemcpyAsync(m, buffer, sizeof(float)*nelem, cudaMemcpyHostToDevice, 0));
	free(buffer);
}

void snudnn_uniform(float *m, size_t nelem, float min, float max)
{
	chkCURAND(curandGenerateUniform(curand_gen, m, nelem));
	cuda_scale(m, nelem, min, max);
}

tensor_t* snudnnEmbeddingForward(tensor_t *x, tensor_t *weights)
{
	return 0;
}


/// -- { tanh
static cudnnActivationDescriptor_t tanh_desc()
{
	static cudnnActivationDescriptor_t desc = NULL;
	if (desc == NULL) {
		chkCUDNN(cudnnCreateActivationDescriptor(&desc));
		chkCUDNN(cudnnSetActivationDescriptor(desc, CUDNN_ACTIVATION_TANH, CUDNN_NOT_PROPAGATE_NAN, 0.0));
	}
	return desc;
}

tensor_t* snudnn_tanh_forward(tensor_t *x)
{
	tensor_t *y = tensor_samesize(x);
	chkCUDNN(cudnnActivationForward(cudnn_handle,
				tanh_desc(),
				&_one, tensor_descriptor(x), tensor_mem(x),
				&_zero, tensor_descriptor(y), tensor_mem(y)));
	return y;
}

tensor_t* snudnn_tanh_backward(tensor_t *dy, tensor_t *x, tensor_t *y)
{
	tensor_t *dx = tensor_samesize(dy);
	chkCUDNN(cudnnActivationBackward(cudnn_handle,
				tanh_desc(),
				&_one, tensor_descriptor(y), tensor_mem(y),
				tensor_descriptor(dy), tensor_mem(dy),
				tensor_descriptor(x), tensor_mem(x),
				&_zero, tensor_descriptor(dx), tensor_mem(dx)));
	return dx;
}
/// -- }

/// -- { softmax
tensor_t* snudnn_softmax0_forward(tensor_t *x)
{
	tensor_t *y = tensor_samesize(x);
	tensor_unsqueeze(x, 0);
	tensor_unsqueeze(y, 0);
	chkCUDNN(cudnnSoftmaxForward(cudnn_handle,
					CUDNN_SOFTMAX_ACCURATE,
					CUDNN_SOFTMAX_MODE_CHANNEL,
					&_one, tensor_descriptor(x), tensor_mem(x),
					&_zero, tensor_descriptor(y), tensor_mem(y)));
	tensor_squeeze(x, 0);
	tensor_squeeze(y, 0);
	return y;
}

tensor_t* snudnn_softmax0_backward(tensor_t *dy, tensor_t *y)
{
	tensor_t *dx = tensor_samesize(dy);
	tensor_unsqueeze(dy, 0);
	tensor_unsqueeze(dx, 0);
	chkCUDNN(cudnnSoftmaxBackward(cudnn_handle,
					CUDNN_SOFTMAX_ACCURATE,
					CUDNN_SOFTMAX_MODE_CHANNEL,
					&_one,
					tensor_descriptor(y), tensor_mem(y), 
					tensor_descriptor(y), tensor_mem(dy),
					&_zero, tensor_descriptor(y), tensor_mem(dx)));
	tensor_squeeze(dy, 0);
	tensor_squeeze(dx, 0);
	return dx;
}
/// -- }
