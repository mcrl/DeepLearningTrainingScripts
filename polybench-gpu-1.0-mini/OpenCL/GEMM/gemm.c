/**
 * gemm.c: This file is part of the PolyBench/GPU 1.0 test suite.
 *
 *
 * Contact: Scott Grauer-Gray <sgrauerg@gmail.com>
 * Louis-Noel Pouchet <pouchet@cse.ohio-state.edu>
 * Web address: http://www.cse.ohio-state.edu/~pouchet/software/polybench/GPU
 */

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <sys/time.h>
#include <math.h>

#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif

#include "../../common/polybenchUtilFuncts.h"

//define the error threshold for the results "not matching"
#define PERCENT_DIFF_ERROR_THRESHOLD 0.05

#define MAX_SOURCE_SIZE (0x100000)

/* Problem size */
#define NI 512
#define NJ 512
#define NK 512

/* Thread block dimensions */
#define DIM_LOCAL_WORK_GROUP_X 32
#define DIM_LOCAL_WORK_GROUP_Y 8

#if defined(cl_khr_fp64)  // Khronos extension available?
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
#elif defined(cl_amd_fp64)  // AMD extension available?
#pragma OPENCL EXTENSION cl_amd_fp64 : enable
#endif

/* Declared constant values for ALPHA and BETA (same as values in PolyBench 2.0) */
#define ALPHA 32412
#define BETA 2123

/* Can switch DATA_TYPE between float and double */
typedef float DATA_TYPE;

char str_temp[1024];

cl_platform_id platform_id;
cl_device_id device_id;   
cl_uint num_devices;
cl_uint num_platforms;
cl_int errcode;
cl_context clGPUContext;
cl_kernel clKernel;
cl_command_queue clCommandQue;
cl_program clProgram;
cl_mem a_mem_obj;
cl_mem b_mem_obj;
cl_mem c_mem_obj;
FILE *fp;
char *source_str;
size_t source_size;



void compareResults(DATA_TYPE* C, DATA_TYPE* C_outputFromGpu)
{
	int i, j, fail;
	fail = 0;
	
	// Compare C1 and C2
	for (i=0; i < NI; i++) 
	{
		for (j=0; j < NJ; j++) 
		{
			if (percentDiff(C[i*NJ + j], C_outputFromGpu[i*NJ + j]) > PERCENT_DIFF_ERROR_THRESHOLD) 
			{
				fail++;
			}
		}
	}
	
	// Print results
	printf("Non-Matching CPU-GPU Outputs Beyond Error Threshold of %4.2f Percent: %d\n", PERCENT_DIFF_ERROR_THRESHOLD, fail);

}


void read_cl_file()
{
	// Load the kernel source code into the array source_str
#ifndef WITH_BINARY
	fp = fopen(WITH_SOURCE, "r");
#else
    fp = fopen(WITH_BINARY, "rb");
#endif
    if (!fp) {
        fprintf(stdout, "Failed to load kernel.\n");
        exit(1);
    }
    fseek(fp, 0, SEEK_END);
    source_size = ftell(fp);
    rewind(fp);
    source_str = (char*)malloc(source_size);
    fread(source_str, 1, source_size, fp);
    fclose(fp);
}


void init(DATA_TYPE *A, DATA_TYPE *B, DATA_TYPE *C)
{
	int i, j;

  	for (i = 0; i < NI; i++)
	{
    	for (j = 0; j < NK; j++)
		{
      		A[i*NK + j] = ((DATA_TYPE) i*j) / NI;
		}
	}

  	for (i = 0; i < NK; i++)
	{
    	for (j = 0; j < NJ; j++)
		{
      		B[i*NJ + j] = ((DATA_TYPE) i*j + 1) / NJ;
		}
	}

  	for (i = 0; i < NI; i++)
	{
    	for (j = 0; j < NJ; j++)
		{
      		C[i*NJ + j] = ((DATA_TYPE) i*j + 2) / NJ;
		}
	}
}


void cl_initialization()
{	
	// Get platform and device information
	errcode = clGetPlatformIDs(1, &platform_id, &num_platforms);
	if(errcode == CL_SUCCESS) printf("number of platforms is %d\n",num_platforms);
	else printf("Error getting platform IDs\n");

	errcode = clGetPlatformInfo(platform_id,CL_PLATFORM_NAME, sizeof(str_temp), str_temp,NULL);
	if(errcode == CL_SUCCESS) printf("platform name is %s\n",str_temp);
	else printf("Error getting platform name\n");

	errcode = clGetPlatformInfo(platform_id, CL_PLATFORM_VERSION, sizeof(str_temp), str_temp,NULL);
	if(errcode == CL_SUCCESS) printf("platform version is %s\n",str_temp);
	else printf("Error getting platform version\n");

	errcode = clGetDeviceIDs( platform_id, CL_DEVICE_TYPE_ALL, 1, &device_id, &num_devices);
	if(errcode == CL_SUCCESS) printf("number of devices is %d\n", num_devices);
	else printf("Error getting device IDs\n");

	errcode = clGetDeviceInfo(device_id,CL_DEVICE_NAME, sizeof(str_temp), str_temp,NULL);
	if(errcode == CL_SUCCESS) printf("device name is %s\n",str_temp);
	else printf("Error getting device name\n");
	
	// Create an OpenCL context
	clGPUContext = clCreateContext( NULL, 1, &device_id, NULL, NULL, &errcode);
	if(errcode != CL_SUCCESS) printf("Error in creating context\n");
 
	//Create a command-queue
	clCommandQue = clCreateCommandQueue(clGPUContext, device_id, 0, &errcode);
	if(errcode != CL_SUCCESS) printf("Error in creating command queue\n");
}


void cl_mem_init(DATA_TYPE* A, DATA_TYPE* B, DATA_TYPE* C)
{
	a_mem_obj = clCreateBuffer(clGPUContext, CL_MEM_READ_WRITE, sizeof(DATA_TYPE) * NI * NK, NULL, &errcode);
	b_mem_obj = clCreateBuffer(clGPUContext, CL_MEM_READ_WRITE, sizeof(DATA_TYPE) * NK * NJ, NULL, &errcode);
	c_mem_obj = clCreateBuffer(clGPUContext, CL_MEM_READ_WRITE, sizeof(DATA_TYPE) * NI * NJ, NULL, &errcode);
		
	if(errcode != CL_SUCCESS) printf("Error in creating buffers\n");

	errcode = clEnqueueWriteBuffer(clCommandQue, a_mem_obj, CL_TRUE, 0, sizeof(DATA_TYPE) * NI * NK, A, 0, NULL, NULL);
	errcode = clEnqueueWriteBuffer(clCommandQue, b_mem_obj, CL_TRUE, 0, sizeof(DATA_TYPE) * NK * NJ, B, 0, NULL, NULL);
	errcode = clEnqueueWriteBuffer(clCommandQue, c_mem_obj, CL_TRUE, 0, sizeof(DATA_TYPE) * NI * NJ, C, 0, NULL, NULL);
	if(errcode != CL_SUCCESS)printf("Error in writing buffers\n");
}


void cl_load_prog()
{
	// Create a program from the kernel source
#ifndef WITH_BINARY
    clProgram = clCreateProgramWithSource(clGPUContext, 1, (const char **)&source_str, (const size_t *)&source_size, &errcode);
#else
    clProgram = clCreateProgramWithBinary(clGPUContext, 1, &device_id, (const size_t *)&source_size, (const unsigned char **)&source_str, NULL, &errcode);
#endif

	if(errcode != CL_SUCCESS) printf("Error in creating program\n");

	// Build the program
	errcode = clBuildProgram(clProgram, 1, &device_id, NULL, NULL, NULL);
	if(errcode != CL_SUCCESS) printf("Error in building program\n");
		
	// Create the OpenCL kernel
	clKernel = clCreateKernel(clProgram, "gemm", &errcode);
	if(errcode != CL_SUCCESS) printf("Error in creating kernel\n");
	clFinish(clCommandQue);
}


void cl_launch_kernel()
{
	double t_start, t_end;

	int ni=NI;
	int nj=NJ;
	int nk=NK;

	DATA_TYPE alpha = ALPHA;
	DATA_TYPE beta = BETA;

	size_t localWorkSize[2], globalWorkSize[2];
	localWorkSize[0] = DIM_LOCAL_WORK_GROUP_X;
	localWorkSize[1] = DIM_LOCAL_WORK_GROUP_Y;
	globalWorkSize[0] = (size_t)ceil(((float)NJ) / ((float)DIM_LOCAL_WORK_GROUP_X)) * DIM_LOCAL_WORK_GROUP_X;
	globalWorkSize[1] = (size_t)ceil(((float)NI) / ((float)DIM_LOCAL_WORK_GROUP_Y)) * DIM_LOCAL_WORK_GROUP_Y;

	t_start = rtclock();
#ifdef MEASURE
measure_start();
do {
#endif
	
	// Set the arguments of the kernel
	errcode =  clSetKernelArg(clKernel, 0, sizeof(cl_mem), (void *)&a_mem_obj);
	errcode |= clSetKernelArg(clKernel, 1, sizeof(cl_mem), (void *)&b_mem_obj);
	errcode |= clSetKernelArg(clKernel, 2, sizeof(cl_mem), (void *)&c_mem_obj);
	errcode |= clSetKernelArg(clKernel, 3, sizeof(DATA_TYPE), (void *)&alpha);
	errcode |= clSetKernelArg(clKernel, 4, sizeof(DATA_TYPE), (void *)&beta);
	errcode |= clSetKernelArg(clKernel, 5, sizeof(int), (void *)&ni);
	errcode |= clSetKernelArg(clKernel, 6, sizeof(int), (void *)&nj);
	errcode |= clSetKernelArg(clKernel, 7, sizeof(int), (void *)&nk);
	
	if(errcode != CL_SUCCESS) printf("Error in seting arguments\n");

	// Execute the OpenCL kernel
	errcode = clEnqueueNDRangeKernel(clCommandQue, clKernel, 2, NULL, globalWorkSize, localWorkSize, 0, NULL, NULL);
	if(errcode != CL_SUCCESS) printf("Error in launching kernel\n");
	clFinish(clCommandQue);

#ifdef MEASURE
} while (measure_continue());
measure_end();
#endif
	t_end = rtclock();
	fprintf(stdout, "GPU Runtime: %0.6lfs\n", t_end - t_start);
}


void cl_clean_up()
{
	// Clean up
	errcode = clFlush(clCommandQue);
	errcode = clFinish(clCommandQue);
	errcode = clReleaseKernel(clKernel);
	errcode = clReleaseProgram(clProgram);
	errcode = clReleaseMemObject(a_mem_obj);
	errcode = clReleaseMemObject(b_mem_obj);
	errcode = clReleaseMemObject(c_mem_obj);
	errcode = clReleaseCommandQueue(clCommandQue);
	errcode = clReleaseContext(clGPUContext);
	if(errcode != CL_SUCCESS) printf("Error in cleanup\n");
}


void gemm(DATA_TYPE *A, DATA_TYPE *B, DATA_TYPE *C)
{
	int i,j,k;
	
	for (i = 0; i < NI; i++)
	{
    		for (j = 0; j < NJ; j++)
    		{
			C[i*NJ + j] *= BETA;
	
			for (k = 0; k < NK; ++k)
			{
	  			C[i*NJ + j] += ALPHA * A[i*NK + k] * B[k*NJ + j];
			}
      		}
	}
}


int main(void) 
{
	double t_start, t_end;

	DATA_TYPE* A;
	DATA_TYPE* B;  
	DATA_TYPE* C;  
	DATA_TYPE* C_outputFromGpu; 

	A = (DATA_TYPE*)malloc(NI*NK*sizeof(DATA_TYPE)); 
	B = (DATA_TYPE*)malloc(NK*NJ*sizeof(DATA_TYPE));   
	C = (DATA_TYPE*)malloc(NI*NJ*sizeof(DATA_TYPE)); 
	C_outputFromGpu = (DATA_TYPE*)malloc(NI*NJ*sizeof(DATA_TYPE)); 

	init(A, B, C);
	read_cl_file();
	cl_initialization();
	cl_mem_init(A, B, C);
	cl_load_prog();

	cl_launch_kernel();
#ifdef MEASURE
  measure_disable();
	cl_mem_init(A, B, C);
	cl_launch_kernel();
#endif

	errcode = clEnqueueReadBuffer(clCommandQue, c_mem_obj, CL_TRUE, 0, NI*NJ*sizeof(DATA_TYPE), C_outputFromGpu, 0, NULL, NULL);
	if(errcode != CL_SUCCESS) printf("Error in reading GPU mem\n");

	t_start = rtclock();
	gemm(A, B, C);
	t_end = rtclock(); 
	fprintf(stdout, "CPU Runtime: %0.6lfs\n", t_end - t_start);   
	compareResults(C, C_outputFromGpu);
	cl_clean_up();

	free(A);
	free(B);  
	free(C);  
	free(C_outputFromGpu); 

	return 0;
}

