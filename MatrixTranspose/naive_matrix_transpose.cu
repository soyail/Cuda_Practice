
#include "naive_matrix_transpose.h"

#define BlockDimX 32
#define BlockDimY 32

__global__ void NaiveKernel(int* input, int* output, int w, int h){
    unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
    if(x<w && y<h){
        output[x*h+y] = input[y*w+x];
    }
}

void naive_matrix_transpose(
    int* input,
    int* output,
    int w,
    int h
){

    dim3 block(BlockDimX, BlockDimY);
    dim3 grid((w+BlockDimX-1)/BlockDimX,(h+BlockDimY-1)/BlockDimY);
    NaiveKernel<<<block, grid>>>(input, output, w, h);
}