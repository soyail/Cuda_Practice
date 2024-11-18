#include "conv2d_img2col_gpu.h"



void img2col_kernel(
    const float* input,
    float* output,
    int input_width,
    int input_height,
    int kernel_size,
    int output_width,
    int output_height,
    int stride
){
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int in_x = x * stride;
    int in_y = y * stride;
    int ksize = kernel_size * kernel_size;
    for(int kh=0; kh<kernel_size; ++kh){
        for(int kw=0; kw<kernel_size; ++kw){
            output[(y*output_width+x)*ksize+kh*kernel_size+kw] = input[(in_y+kh)*input_width+kw];
        }
    }
}

void conv2d_img2col_gemm(
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
    int output_size = output_width*output_height;
    int ksize = kernel_size*kernel_size;
    float* d_output;
    cudaMalloc(&d_output, output_size*ksize*sizeof(float));

    // cuda
    dim3 blockDim(16,16);
    dim3 gridDim((output_width+blockDim.x-1)/blockDim.x, (output_height+blockDim.y-1)/blockDim.y);
    img2col_kernel<<<gridDim, blockDim>>>(input, d_output, input_width, input_height, kernel_size, output_width, output_height, stride);

    cublasHandle_t handle;
    cublasCreate(&handle);
    cublasgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, output_size, 1, ksize, 1.0f, d_output, ksize, kernel, 1, 0.0f, output, 1);
    
    cublasDestroy(handle);
    cudaFree(d_output);
}