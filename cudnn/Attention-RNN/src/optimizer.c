#include <math.h>
#include <nccl.h>
#include "utils.h"
#include "snudnn.h"
#include "gpu.h"
#include "optimizer.h"

static double lr = 1.0;
static double clip_norm = 1.0;
//static double momentum = 0.99;
void optimizer_step(const void *dw, void *w, size_t nelem, size_t N)
{
	chkNCCL(ncclAllReduce((void*)dw, (void*)dw, nelem, ncclFloat, ncclSum, comm, update_stream));
	chkCUBLAS(cublasSetStream(cublas_handle, update_stream));
	float nrm2 = 0.0f;

	float scale = 1.0 / N;
	chkCUBLAS(cublasSscal(cublas_handle, nelem, &scale, (float*)dw, 1));
	chkCUBLAS(cublasSnrm2(cublas_handle, nelem, (float*)dw, 1, &nrm2));
	if (nrm2 > clip_norm) {
		float scale = sqrt(clip_norm)/nrm2;
		chkCUBLAS(cublasSscal(cublas_handle, nelem, &scale, (float*)dw, 1));
	}

	float neg_lr = -lr;
	chkCUBLAS(cublasSaxpy(cublas_handle, nelem, &neg_lr, (float*)dw, 1, w, 1));
	chkCUBLAS(cublasSetStream(cublas_handle, data_stream));
}
