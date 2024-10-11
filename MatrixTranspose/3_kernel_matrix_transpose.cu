#include "3_kernel_matrix_transpose.h"


#define BlockDimX 32
#define BlockDimY 32
__global__ void UnrollKernel(int* input, int* output, int w, int h) {
	unsigned int x = 2 * blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;

	__shared__ int tile[BlockDimY][2 * BlockDimX + 1];
	unsigned int read = x + y * w;

	if (x < w && y < h) {
		tile[threadIdx.y][threadIdx.x] = input[read];
	}
	if (x + blockDim.x < w && y < h) {
		tile[threadIdx.y][threadIdx.x + BlockDimX] = input[read + BlockDimX];
	}
	__syncthreads();

	//输入矩阵block线性索引
	unsigned int idx_block = threadIdx.y * blockDim.x + threadIdx.x;

	//转置矩阵block索引映射
	unsigned int x_map = idx_block % blockDim.y;
	unsigned int y_map = idx_block / blockDim.y;

	//转置矩阵全局索引
	x = blockIdx.y * blockDim.y + x_map;
	y = 2 * blockIdx.x * blockDim.x + y_map;
	unsigned int write = y * h + x;

	if (x < h && y < w) {
		output[write] = tile[x_map][y_map];
	}
	if (x < h && y + blockDim.x < w) {
		output[write + h * BlockDimX] = tile[x_map][y_map + BlockDimX];
	}
}


void matrix_transpose_unroll(
    int* input,
    int* output,
    int w,
    int h
){
    dim3 block(BlockDimX, BlockDimY);
    //注意这里的grid.x
	dim3 grid((w + 2 * BlockDimX - 1) / (2 * BlockDimX), (h + BlockDimY - 1) / BlockDimY);
    UnrollKernel << <grid, block >> > (input, output, w, h);
}