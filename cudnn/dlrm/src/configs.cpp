#include "configs.h"

using namespace std;


// these configurations should be modified at the start of the main
std::vector<int> botFCLayers = {13, 512, 256, 64, 16};
std::vector<int> topFCLayers = {367, 512, 256, 1};

int batch_size = 131072, epochs = 500;
int num_sparse = 26, num_dense = 13;
int vector_size = 16, bag_size = 4;

// int train_batches = 1, test_batches = 1;   // profile
// int train_batches = 100, test_batches = 100;   // test
// int train_batches = 10000, test_batches = 5000;   // small training
// int train_batches = 50000, test_batches = 10000; // medium training
// int train_batches = 300000, test_batches = 50000; // full training
int train_batches = 300000 * 128 / batch_size, test_batches = 50000 * 128 / batch_size; // full training

float one = 1.0, zero = 0.0, minusone = -1.0;
float lr = -1.0 / batch_size; 
// bool inference_only = true;
bool inference_only = false;

int NDEV = 1;
int NNODE = 1;
int hostdev = 0;
int hostnode = 0;
int USEBAG = 0;


/***************************************************************/
/*                          CUDA                               */
/***************************************************************/

const char *cublasGetErrorString(cublasStatus_t status)
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


cudaError_t cudaStat;
cudnnHandle_t cudnn[MAXDEV];
cublasHandle_t cublas[MAXDEV];
cudaStream_t streams[MAXDEV];

void cuda_init () {
    int deviceCount = 0;
    cudaGetDeviceCount(&deviceCount);
    for (int ndev = 0; ndev < NDEV; ndev++) {
        CUDA_CALL( cudaSetDevice(ndev) );
        CUDA_CALL( cudaStreamCreate( &streams[ndev] ) );
        CUDNN_CALL( cudnnCreate(&cudnn[ndev]) );
        CUBLAS_CALL( cublasCreate(&cublas[ndev]) );
    }
}

/***************************************************************/
/*                          MPI                                */
/***************************************************************/

int mpi_world_size;
int mpi_world_rank;
void mpi_init () {
    int t;
    MPI_Init_thread(NULL, NULL, MPI_THREAD_FUNNELED, &t);
    MPI_Comm_size(MPI_COMM_WORLD, &mpi_world_size);

    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_world_rank);
    // Get the name of the processor
    char processor_name[MPI_MAX_PROCESSOR_NAME];
    int name_len;
    MPI_Get_processor_name(processor_name, &name_len);

    // Print off a hello world message
    printf("Hello world from processor %s, rank %d out of %d processors\n",
           processor_name, mpi_world_rank, mpi_world_size);
}

bool is_host () {
    return mpi_world_rank == hostnode;
}

/***************************************************************/
/*                          OpenMP                             */
/***************************************************************/

void omp_init () {
    omp_set_num_threads(NDEV);
}

/***************************************************************/
/*                          NCCL                               */
/***************************************************************/


ncclUniqueId nccl_id;
ncclComm_t comms[MAXDEV];
int devlist[MAXDEV];

void nccl_init () {
    if ( mpi_world_rank == 0 ) ncclGetUniqueId(&nccl_id);
    MPI_Bcast(&nccl_id, NCCL_UNIQUE_ID_BYTES, MPI_CHAR, 0, MPI_COMM_WORLD);

    for (int i = 0; i < NDEV; i++) devlist[i] = i;

    #pragma omp parallel
    {
        int ndev = omp_get_thread_num();
        CUDA_CALL( cudaSetDevice(ndev) );
        NCCL_CALL( ncclCommInitRank(&comms[ndev], mpi_world_size * NDEV, nccl_id, mpi_world_rank * NDEV + ndev) );
    }
}
