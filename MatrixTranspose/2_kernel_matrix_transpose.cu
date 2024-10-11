#include "2_kernel_matrix_transpose.h"

#define BlockDimX 32
#define BlockDimY 32
#define NUM_PER_THREAD 4


__global__ void matrix_transpose_threadtiling_kernel(int* input, int* output, int w, int h){
    // input[h,w]
    __shared__ int tile[BlockDimY][BlockDimX*NUM_PER_THREAD];
    // input
    input += blockIdx.y * blockDim.y * w + blockIdx.x * blockDim.x * NUM_PER_THREAD;
    
    for(unsigned int i=0; i<BlockDimX*NUM_PER_THREAD; i+=BlockDimX){
        tile[threadIdx.y][threadIdx.x+i] = input[threadIdx.y*w+threadIdx.x+i];
    }
    __syncthreads();
    // output[w,h]
    output += blockIdx.x * blockDim.x * NUM_PER_THREAD * h + blockIdx.y * blockDim.y;
    for(unsigned int i=0; i<BlockDimX*NUM_PER_THREAD; i+=BlockDimX){
        output[(threadIdx.y+i)*h+threadIdx.x] = tile[threadIdx.x][threadIdx.y+i];
    }
}


void matrix_transpose_threadtiling(
    int* input,
    int* output,
    int w,
    int h
){
    dim3 block(BlockDimX, BlockDimY);
    dim3 grid((w+BlockDimX*NUM_PER_THREAD-1)/(BlockDimX*NUM_PER_THREAD),(h+BlockDimY-1)/(BlockDimY));
    matrix_transpose_threadtiling_kernel<<<block, grid>>>(input, output, w, h);
}