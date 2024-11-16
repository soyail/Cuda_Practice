#include <random>
#include <iostream>
#include <cuda_runtime.h>
#include "conv2d_cpu.h"
#include "conv2d_naive.h"
#include "conv2d_v1.h"
#include "conv2d_img2col.h"

typedef void (*impl_t){
    const float* input,
    const float* kernel,
    float* output,
    int input_width,
    int input_height,
    int kernel_size,
    int stride,
    int output_width,
    int output_height
};

struct impl{
    std::string name;
    impl_t impl;
    bool is_gpu;
}

std::vector<impl> conv_impls = {
    {"conv2d_naive", conv2d_naive, true},
    {"conv2d_img2col", conv2d_img2col_gemm, false}
};

void verify_result(const float* gpu_output, const float* cpu_output, int size) {
    for (int i = 0; i < size; ++i) {
        if (fabs(gpu_output[i] - cpu_output[i]) > 1e-5) {
            std::cout << "Mismatch at index " << i << ": GPU: " << gpu_output[i]
                      << ", CPU: " << cpu_output[i] << std::endl;
            return;
        }
    }
    std::cout << "Results match!" << std::endl;
}




int main(){
    int input_width = 256;
    int input_height = 256;
    int kernel_size = 5;
    float* input = new float[input_width*input_height];
    float* kernel = new float[kernel_size*kernel_size];

    for(int i=0; i<input_width; ++i){
        for(int j=0; j<input_height; ++j){
            input[input_width*i+j] = static_cast<float>(rand())/RAND_MAX;
        }
    }
    for(int i=0; i<kernel_size; ++i){
        for(int j=0; j<kernel_size; ++j){
            kernel[i*kernel_size+j] = static_cast<float>(rand())/RAND_MAX;
        }
    }
    int stride = 1;
    int output_width = (input_width - kernel_size)/stride + 1; 
    int output_height = (input_height - kernel_size)/stride + 1; 
    float* output = new float[output_height*output_width];
    float* h_output = new float[output_height*output_width];

    float *d_input, *d_kernel, *d_output;
    cudaMalloc(&d_input, input_width * input_height * sizeof(float));
    cudaMalloc(&d_kernel, kernel_size * kernel_size * sizeof(float));
    cudaMalloc(&d_output, output_height * output_width * sizeof(float));

    cudaMemcpy(d_input, input, input_height*input_width*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_kernel, kernel, kernel_size*kernel_size*sizeof(float), cudaMemcpyHostToDevice);

    conv2d_cpu(input, kernel, h_output, input_width, input_height, kernel_size, stride, output_width, output_height);

    for(auto conv_impl:conv_impls){
        std::cout << "testing " << conv_impl.name << std::endl;
        if(conv_impl.is_gpu){
            cudaMemset(d_input, 0.0, output_width*output_height*sizeof(float));
            conv_impl.impl(d_input, d_kernel, d_output, input_width, input_height, kernel_size, stride, output_width, output_height);
            cudaError_t err = cudaGetLastError();
            if(err != cudaSuccess){
                printf("%s CUDA Kernel Launch Error: %s\n", conv_impl.name, cudaGetErrorString(err));
            }
            cudaDeviceSynchronize();

            cudaMemcpy(output, d_output, output_width*output_height*sizeof(float), cudaMemcpyDeviceToHost);

            verify_result(h_output, output, output_width*output_height);
        }else{
            Memset(h_output, 0,0, output_width*output_height*sizeof(float));
            conv_impl.impl(input, kernel, h_output, input_width, input_height, kernel_size, stride, output_width, output_height);
            verify_result(h_output, output, output_width*output_height);
        }
    }
    


    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_kernel);
    delete[] input;
    delete[] output;
    delete[] kernel;
    return 0;
}

