#include <cuda_runtime.h>
#include "conv2d_naive.h"
#include <iostream>

__global__ void conv2d_v1_kernel(
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
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    // 定义共享内存大小
    int shared_input_width = kernel_size + (blockDim.x - 1) * stride;
    int shared_input_height = kernel_size + (blockDim.y - 1) * stride;
    __shared__ float shared_kernel[25];  // 假设最大 kernel 大小为 5x5
    extern __shared__ float shared_input[];

    // 加载卷积核到共享内存
    if (tx < kernel_size && ty < kernel_size) {
        shared_kernel[ty * kernel_size + tx] = kernel[ty * kernel_size + tx];
    }

    // 计算每个线程块所覆盖的输入区域
    int input_x = x * stride - kernel_size / 2 + tx;
    int input_y = y * stride - kernel_size / 2 + ty;

    // 计算每个线程需要加载的元素数量
    int numsPerThread = shared_input_width * shared_input_height / (blockDim.x * blockDim.y) + 1;

    // 每个线程加载多个元素
    for (int i = 0; i < numsPerThread; i++) {
        int idx_x = input_x + (i % blockDim.x);
        int idx_y = input_y + (i / blockDim.x);

        // 确保在合法范围内进行数据加载
        if (idx_x >= 0 && idx_x < input_width && idx_y >= 0 && idx_y < input_height) {
            int shared_x = tx + (i % blockDim.x);
            int shared_y = ty + (i / blockDim.x);
            shared_input[shared_y * shared_input_width + shared_x] = input[idx_y * input_width + idx_x];
        }
    }

    __syncthreads(); // 同步线程，确保共享内存加载完成

    // 执行卷积计算
    if (x < output_width && y < output_height) {
        float result = 0.0f;
        for (int i = 0; i < kernel_size; ++i) {
            for (int j = 0; j < kernel_size; ++j) {
                int shared_x = tx + j;
                int shared_y = ty + i;
                result += shared_kernel[i * kernel_size + j] * shared_input[shared_y * shared_input_width + shared_x];
            }
        }
        output[y * output_width + x] = result;
    }
}

void conv2d_v1(
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
    dim3 blockDim(16, 16);
    dim3 gridDim((output_width + blockDim.x - 1) / blockDim.x, (output_height + blockDim.y - 1) / blockDim.y);
    
    int shared_input_size = (kernel_size + (blockDim.x - 1) * stride) * (kernel_size + (blockDim.y - 1) * stride);
    conv2d_v1_kernel<<<gridDim, blockDim, shared_input_size * sizeof(float)>>>(input, kernel, output, input_width, input_height, kernel_size, stride, output_width, output_height);
}
