
#include <cuda_runtime.h>
#include "conv2d_naive.h"
#include <iostream>

__global__ void conv2d_kernel(
    const float* input, 
    const float* kernel, 
    float* output,
    int input_width, 
    int input_height, 
    int kernel_size,
    int stride,
    int output_width,
    int output_height
){
    int x = blockDim.x * blockIdx.x + threadIdx.x;
    int y = blockDim.y * blockIdx.y + threadIdx.y;
    if(x<output_width && y<output_height){
        float result = 0.0f;
        int input_x = x * stride;
        int input_y = y * stride;
        for(int i=0; i<kernel_size; ++i){
            for(int j=0; j<kernel_size; ++j){
                int in_x = input_x+i;
                int in_y = input_y+j;
                if(in_x>=0 && in_x<input_width && in_y>=0 && in_y<input_height){
                    result += kernel[j*kernel_size+i] * input[in_y*input_width+in_x]; 
                }
            }
        }
        output[y*output_width+x] = result;
    }
}

void conv2d_naive(
    const float* input,
    const float* kernel,
    float* output,
    int input_width,
    int input_height,
    int kernel_size,
    int stride,
    int output_width,
    int output_height
){
    dim3 blockDim(16,16);
    dim3 gridDim((output_width+blockDim.x-1)/blockDim.x, (output_height+blockDim.y-1)/blockDim.y);
    conv2d_kernel<<<gridDim, blockDim>>>(input, kernel, output, input_width, input_height, kernel_size, stride, output_width, output_height);

}