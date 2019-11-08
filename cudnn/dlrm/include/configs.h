#ifndef _CONFIGS_H_
#define _CONFIGS_H_

#include <bits/stdc++.h>


#include <cuda.h>
#include <cudnn.h>
#include <cublas_v2.h>
#include <nccl.h>
#include <mpi.h>
#include <omp.h>

using namespace std;

extern std::vector<int> botFCLayers;
extern std::vector<int> topFCLayers ;

extern int batch_size, epochs;
extern int num_sparse, num_dense;
extern int vector_size, bag_size;

extern int train_batches, test_batches;

extern float one, zero, minusone;
extern float lr; // this should be modified at start of the main

extern bool inference_only;

#define MAXDEV 4
#define MAXNODE 8
extern int NDEV;
extern int NNODE;
extern int hostdev;
extern int hostnode;

#define DEBUG 1
extern int USEBAG;

/***************************************************************/
/*                          CUDA                               */
/***************************************************************/

#define CUDA_CALL(f) { \
    cudaError_t err = (f); \
    if (err != cudaSuccess) { \
        cerr \
            << "    Error at [" << __FILE__ << ":" << __LINE__ << "] " << cudaGetErrorString(err) << endl; \
        exit(1); \
    } \
  }

#define CUDNN_CALL(f) { \
    cudnnStatus_t err = (f); \
    if (err != CUDNN_STATUS_SUCCESS) { \
        cerr \
            << "    Error at [" << __FILE__ << ":" << __LINE__ << "] "\
            << cudnnGetErrorString(err) << endl; \
        exit(1); \
    } \
  }


#define CUBLAS_CALL(f) { \
    cublasStatus_t  err = (f); \
    if (err != CUBLAS_STATUS_SUCCESS) { \
        cerr \
            << "    Error at [" << __FILE__ << ":" << __LINE__ << "] " << cublasGetErrorString(err) << endl; \
        exit(1); \
    } \
  }

const char *cublasGetErrorString(cublasStatus_t status);



extern cudaError_t cudaStat;
extern cudnnHandle_t cudnn[MAXDEV];
extern cublasHandle_t cublas[MAXDEV];
extern cudaStream_t streams[MAXDEV];

void cuda_init ();

/***************************************************************/
/*                          MPI                                */
/***************************************************************/


extern int mpi_world_size;
extern int mpi_world_rank;
void mpi_init ();
bool is_host ();

/***************************************************************/
/*                          OpenMP                             */
/***************************************************************/

void omp_init ();

/***************************************************************/
/*                          NCCL                               */
/***************************************************************/


#define NCCL_CALL(cmd) do {                         \
  ncclResult_t r = cmd;                             \
  if (r!= ncclSuccess) {                            \
    printf("Failed, NCCL error %s:%d '%s'\n",             \
        __FILE__,__LINE__,ncclGetErrorString(r));   \
    exit(EXIT_FAILURE);                             \
  }                                                 \
} while(0)


extern ncclUniqueId nccl_id;
extern ncclComm_t comms[MAXDEV];
extern int devlist[MAXDEV];

void nccl_init ();

#endif // _CONFIGS_H_
