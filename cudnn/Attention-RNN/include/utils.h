#ifndef __UTILS_H__
#define __UTILS_H__

#include <time.h>
#include <sys/time.h>
#include <curand.h>
#include "gpu.h"
#include "snudnn.h"

//#define __DEBUG__

#define START_TIMER                                            \
  {                                                            \
    struct timespec st;                                        \
    chkCUDA(cudaDeviceSynchronize());                          \
    clock_gettime(CLOCK_MONOTONIC, &st);

#define STOP_TIMER(tg)                                         \
    struct timespec ed;                                        \
    chkCUDA(cudaDeviceSynchronize());                          \
    clock_gettime(CLOCK_MONOTONIC, &ed);                       \
	  if (rank == 0)                                          \
	  fprintf(stderr, "%s: %fms\n", tg, (ed.tv_sec - st.tv_sec) * 1000 + (ed.tv_nsec - st.tv_nsec) / 1000000.0); \
  }


#ifdef __DEBUG__
#define START_TIMER_    START_TIMER
#define STOP_TIMER_(tg) STOP_TIMER(tg)
#else

#define START_TIMER_
#define STOP_TIMER_(tg)
#endif

#define chkMPI(cmd) do {                          \
	int e = cmd;                                      \
	if( e != MPI_SUCCESS ) {                          \
		printf("Failed: MPI error %s:%d '%d'\n",        \
				__FILE__,__LINE__, e);   \
		exit(EXIT_FAILURE);                             \
	}                                                 \
} while(0)

#define chkNCCL(cmd) do {                         \
	ncclResult_t r = cmd;                             \
	if (r!= ncclSuccess) {                            \
		printf("Failed, NCCL error %s:%d '%s'\n",             \
				__FILE__,__LINE__,ncclGetErrorString(r));   \
		exit(EXIT_FAILURE);                             \
	}                                                 \
} while(0)

#define chkCURAND(cmd) do {                         \
	curandStatus_t r = cmd;                             \
	if (r!= CURAND_STATUS_SUCCESS) {                            \
		printf("Failed, CURAND error %s:%d '%s'\n",             \
				__FILE__,__LINE__,ncclGetErrorString(r));   \
		exit(EXIT_FAILURE);                             \
	}                                                 \
} while(0)

#endif //__UTILS_H__
