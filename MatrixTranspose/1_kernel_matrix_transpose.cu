#include "1_kernel_matrix_transpose.h"


#define BlockDimX 32
#define BlockDimY 32

__global__ void matrix_transpose_sm_kernel(int* input, int* output, int w, int h){
    // input[h,w]
    input += blockIdx.y * blockDim.y * w + blockIdx.x * blockDim.x;
    __shared__ int tile[BlockDimY][BlockDimX+1];

    tile[threadIdx.y][threadIdx.x] = input[threadIdx.y*w+threadIdx.x];
    __syncthreads();
    // output[w,h]
    output += blockIdx.x * blockDim.x * h + blockIdx.y * blockDim.y;
    output[threadIdx.y*h+threadIdx.x] = tile[threadIdx.x][threadIdx.y];
}



void matrix_transpose_sm(
    int* input,
    int* output,
    int w,
    int h
){
    dim3 block(BlockDimX, BlockDimY);
    dim3 grid((w+BlockDimX-1)/BlockDimX,(h+BlockDimY-1)/BlockDimY);
    matrix_transpose_sm_kernel<<<block, grid>>>(input, output, w, h);
};