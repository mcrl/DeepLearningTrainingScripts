#include <cuda.h>
#include <cuda_runtime_api.h>
#include "kernel.h"
#include <stdio.h>

#define CDIV(a,b) (((a) + (b) - 1) / (b))

__global__ void shuffle(float *in, float *out, int N, int G, int C, int HW, int output_n_offset)
{
	int n = blockIdx.x * blockDim.x + threadIdx.x;
	int g = blockIdx.y * blockDim.y + threadIdx.y;
	int c = blockIdx.z * blockDim.z + threadIdx.z;

	if (n >= N || g >= G || c >= C) {
		return;
	}

	for (int i = 0; i < HW; ++i) {
		out[(n * output_n_offset) + ((c * G) + g) * HW + i] = in[(n * output_n_offset)+((g * C) +c) * HW + i];
	}
}

__global__ void concatenate(float *in, float *out, int N, int C, int HW, int output_n_offset, int base_offset)
{
	int n  = blockIdx.x * blockDim.x + threadIdx.x;
	int c  = blockIdx.y * blockDim.y + threadIdx.y;
	int hw = blockIdx.z * blockDim.z + threadIdx.z;

	if (n >= N || c >= C || hw >= HW) {
		return;
	}
	out[ (n * output_n_offset) + (c * HW) + hw + base_offset ] = in[ (n*C*HW) + (c * HW) + hw];
}

__global__ void pad(float *in, float *out, int N, int C, int srcH, int srcW, int targetW, int srcHW, int targetHW)
{
	int c  = blockIdx.x * blockDim.x + threadIdx.x;
	int h  = blockIdx.y * blockDim.y + threadIdx.y;
	int w  = blockIdx.z * blockDim.z + threadIdx.z;

	if (c >= C || h >= srcH || w >= srcW) {
		return;
	}
	out[ (c * targetHW) + (h * targetW) + w] = in[ (c * srcHW) + (h * srcW) + w ];
}

__global__ void decode_boxes(float *rel_codes, float *boxes, float *pred_boxes, int N, int C, int HW)
{
	int n  = blockIdx.x * blockDim.x + threadIdx.x;
	int c  = blockIdx.y * blockDim.y + threadIdx.y;
	if (n >= N || c >= C) {
		return;
	}

	int wx = 10;
	int wy = 10;
	int ww = 5;
	int wh = 5;

	float boxes_x1 = boxes[(n * C) + (c * HW) + 1];
	float boxes_y1 = boxes[(n * C) + (c * HW) + 0];
	float boxes_x2 = boxes[(n * C) + (c * HW) + 3];
	float boxes_y2 = boxes[(n * C) + (c * HW) + 2];

	float dx = rel_codes[(n * C) + (c * HW) + 1];
	float dy = rel_codes[(n * C) + (c * HW) + 0];
	float dw = rel_codes[(n * C) + (c * HW) + 3];
	float dh = rel_codes[(n * C) + (c * HW) + 2];

	float width = boxes_x2 - boxes_x1;
	float height = boxes_y2 - boxes_y1;
	float ctr_x = boxes_x1 + 0.5 * width;
	float ctr_y = boxes_y1 + 0.5 * height;

	dx = dx / wx;
	dy = dy / wy;
	dw = dw / ww;
	dh = dh / wh;

	float pred_ctr_x = dx * width + ctr_x;
	float pred_ctr_y = dy * height + ctr_y;
	float pred_w = expf(dw) * width;
	float pred_h = expf(dh) * height;


	pred_boxes[(n * C) + (c * HW) + 0] = pred_ctr_x - 0.5 * pred_w;
	pred_boxes[(n * C) + (c * HW) + 1] = pred_ctr_y - 0.5 * pred_h;
	pred_boxes[(n * C) + (c * HW) + 2] = pred_ctr_x + 0.5 * pred_w;
	pred_boxes[(n * C) + (c * HW) + 3] = pred_ctr_y + 0.5 * pred_h;

	return;
}

__global__ void prelu(float *in, float *out, float *sigma, int N, int C, int HW)
{
	int n  = blockIdx.x * blockDim.x + threadIdx.x;
	int c  = blockIdx.y * blockDim.y + threadIdx.y;
	int hw = blockIdx.z * blockDim.z + threadIdx.z;

	if (n >= N || c >= C || hw >= HW) {
		return;
	}

	int offset = (n * (HW * C)) + (c * HW) + hw;
	float val = in[offset];
	
	if (val < 0)
		out[offset] = val * sigma[c];
	else
		out[offset] = val;
	return;
}

void cuda_shuffle(float *in, float *out, int N, int G, int C, int HW)
{                                                                         
	dim3 size_block(8, 8, 8);                                               
	dim3 size_grid(CDIV(N, size_block.x), CDIV(G, size_block.y), CDIV(C, size_block.z)); 
	int output_n_offset = HW * C * G;
	shuffle<<<size_grid, size_block>>>(in, out, N, G, C, HW, output_n_offset);
	return;
}                                                                         

void cuda_concatenate(float *in, float *in2, float *out, int N, int C, int C2, int HW)
{
	dim3 size_block(8, 8, 8);                                               
	dim3 size_grid(CDIV(N, size_block.x), CDIV(C, size_block.y), CDIV(HW, size_block.z)); 

	int output_n_offset = HW * (C + C2);
	int base_offset = 0;
	concatenate<<<size_grid, size_block>>>(in, out, N, C, HW, output_n_offset, base_offset);

	size_grid.y = CDIV(C2, size_block.y);
	base_offset =  HW * C;
	concatenate<<<size_grid, size_block>>>(in2, out, N, C2, HW, output_n_offset, base_offset);
	return;
}

void cuda_pad(float *in, float *out, int N, int C, int srcH, int srcW, int targetH, int targetW)
{
	dim3 size_block(8, 8, 8);
	dim3 size_grid(CDIV(C, size_block.x), CDIV(srcH, size_block.y), CDIV(srcW, size_block.z)); 

	int srcHW = srcH * srcW;
	int targetHW = targetH * targetW;
	pad<<<size_grid, size_block>>>(in, out, N, C, srcH,  srcW, targetW, srcHW, targetHW);
	return;
}

void cuda_decode_boxes(float *rel_codes, float *boxes, float *pred_boxes, int N, int C, int HW)
{
	dim3 size_block(8, 8);
	dim3 size_grid(CDIV(N, size_block.x), CDIV(C, size_block.y));
	decode_boxes<<<size_grid, size_block>>>(rel_codes, boxes, pred_boxes, N, C, HW);
	return;
}

void cuda_prelu(float *in, float* out, float *sigma, int N, int C, int HW)
{
	dim3 size_block(8, 8, 8);                                               
	dim3 size_grid(CDIV(N, size_block.x), CDIV(C, size_block.y), CDIV(HW, size_block.z)); 

	prelu<<<size_grid, size_block>>>(in, out, sigma, N, C, HW);
	return;
}

/* ToDo: shuold rewrite these kernels */
__global__ void depthwise_conv3(float *in, float *out, float *filter, float *bias, int C, int srcH, int srcW, int dstH, int dstW, int stride, int offset)
{
	int c = blockIdx.x * blockDim.x + threadIdx.x;
	int h = blockIdx.y * blockDim.y + threadIdx.y;
	int w = blockIdx.z * blockDim.z + threadIdx.z;

	if (c >= C || h >= dstH || w >= dstW) {
		return;
	}
	//filter size 3(FIXED)
	float filter_1 = filter[ c * (3 * 3) + 0];
	float filter_2 = filter[ c * (3 * 3) + 1];
	float filter_3 = filter[ c * (3 * 3) + 2];
	float filter_4 = filter[ c * (3 * 3) + 3];
	float filter_5 = filter[ c * (3 * 3) + 4];
	float filter_6 = filter[ c * (3 * 3) + 5];
	float filter_7 = filter[ c * (3 * 3) + 6];
	float filter_8 = filter[ c * (3 * 3) + 7];
	float filter_9 = filter[ c * (3 * 3) + 8];

	float bias_val = 0;
	if(bias != NULL) {
		bias_val = bias[c];
	}

	//src
	float src_1 = 0;float src_2 = 0;float src_3 = 0;
	float src_4 = 0;float src_5 = 0;float src_6 = 0;
	float src_7 = 0;float src_8 = 0;float src_9 = 0;

	int src_pos_h = h * stride + offset;
	int src_pos_w = w * stride + offset;
	src_5 = in[ c * (srcH * srcW) + (src_pos_h * srcW) + src_pos_w];
	int posH = src_pos_h - 1; int posW = src_pos_w - 1;	if(!(posH < 0 || posH >= srcH || posW < 0 || posW >= srcW))		src_1 = in[ c * (srcH * srcW) + (posH * srcW) + posW];
	posH = src_pos_h - 1; posW = src_pos_w + 0;			if(!(posH < 0 || posH >= srcH || posW < 0 || posW >= srcW)) 	src_2 = in[ c * (srcH * srcW) + (posH * srcW) + posW];
	posH = src_pos_h - 1; posW = src_pos_w + 1;			if(!(posH < 0 || posH >= srcH || posW < 0 || posW >= srcW)) 	src_3 = in[ c * (srcH * srcW) + (posH * srcW) + posW];
	posH = src_pos_h + 0; posW = src_pos_w - 1;			if(!(posH < 0 || posH >= srcH || posW < 0 || posW >= srcW)) 	src_4 = in[ c * (srcH * srcW) + (posH * srcW) + posW];
	posH = src_pos_h + 0; posW = src_pos_w + 1;			if(!(posH < 0 || posH >= srcH || posW < 0 || posW >= srcW)) 	src_6 = in[ c * (srcH * srcW) + (posH * srcW) + posW];
	posH = src_pos_h + 1; posW = src_pos_w - 1;			if(!(posH < 0 || posH >= srcH || posW < 0 || posW >= srcW)) 	src_7 = in[ c * (srcH * srcW) + (posH * srcW) + posW];
	posH = src_pos_h + 1; posW = src_pos_w + 0;			if(!(posH < 0 || posH >= srcH || posW < 0 || posW >= srcW)) 	src_8 = in[ c * (srcH * srcW) + (posH * srcW) + posW];
	posH = src_pos_h + 1; posW = src_pos_w + 1;			if(!(posH < 0 || posH >= srcH || posW < 0 || posW >= srcW)) 	src_9 = in[ c * (srcH * srcW) + (posH * srcW) + posW];

	float sum = (filter_1 * src_1) + 
			(filter_2 * src_2) +
			(filter_3 * src_3) +
			(filter_4 * src_4) +
			(filter_5 * src_5) +
			(filter_6 * src_6) +
			(filter_7 * src_7) +
			(filter_8 * src_8) +
			(filter_9 * src_9);
	out[c * (dstH * dstW) + (h * dstW) + w] = sum + bias_val;
	return;
}

__global__ void depthwise_conv1(float *in, float *out, float *filter, float *bias, int C, int srcH, int srcW, int dstH, int dstW, int stride, int offset)
{
	int c = blockIdx.x * blockDim.x + threadIdx.x;
	int h = blockIdx.y * blockDim.y + threadIdx.y;
	int w = blockIdx.z * blockDim.z + threadIdx.z;

	if (c >= C || h >= dstH || w >= dstW) {
		return;
	}
	float filter_val = filter[c];
	float bias_val = 0;
	if(bias != NULL) {
		bias_val = bias[c];
	}
	//int src_pos_h = h * stride + offset;
	//int src_pos_w = w * stride + offset;
	float src = in[ c * (srcH * srcW) + (h * srcW) + w];
	float  sum = filter_val * src;
	out[c * (dstH * dstW) + (h * dstW) + w] = sum + bias_val;
	return;
}

void cuda_depthwise_conv(float *in, float *out, float *filter, float *bias, int C, int srcH, int srcW, int dstH, int dstW, int stride, int offset, int filter_size)
{
	dim3 size_block(8, 8, 8);
	dim3 size_grid(CDIV(C, size_block.x), CDIV(dstH, size_block.y), CDIV(dstW, size_block.z));
	if(filter_size == 3)
		depthwise_conv3<<<size_grid, size_block>>>(in, out, filter, bias, C, srcH, srcW, dstH, dstW, stride, offset);
	else if(filter_size == 1) {
		depthwise_conv1<<<size_grid, size_block>>>(in, out, filter, bias, C, srcH, srcW, dstH, dstW, stride, offset);
	}
}
