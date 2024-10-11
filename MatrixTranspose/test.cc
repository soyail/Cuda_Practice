#include <random>
#include <cuda_runtime.h>
#include "naive_matrix_transpose.h"
#include "1_kernel_matrix_transpose.h"
#include "2_kernel_matrix_transpose.h"
#include "3_kernel_matrix_transpose.h"

void run_kernel(int kernel_num, int* input, int* output, int w, int h){
    switch (kernel_num)
    {
    case 0:
        naive_matrix_transpose(input, output, w, h);
        break;
    
    case 1:
        matrix_transpose_sm(input, output, w, h);
        break;
    case 2:
        matrix_transpose_threadtiling(input, output, w, h);
        break;
    case 3:
        matrix_transpose_unroll(input, output, w, h);
        break;
    default:
        break;
    }

}

int main(int argc, char* argv[]){
    int kernel_num = atoi(argv[1]);
    int w = 1024;
    int h = 1024;
    int* input = new int[w*h];
    int* output = new int[w*h];
    int* output_ref = new int[w*h];
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> distrib(-1000,1000);
    for(int i=0; i<w*h; ++i){
        input[i] = distrib(gen);
    }
    int *input_gpu, *output_gpu;
    cudaMalloc(&input_gpu, w*h*sizeof(int));
    cudaMalloc(&output_gpu, w*h*sizeof(int));
    cudaMemcpy(input_gpu, input, w*h*sizeof(int), cudaMemcpyHostToDevice);
    run_kernel(0, input_gpu, output_gpu, w, h);
    cudaMemcpy(output_ref, output_gpu, w*h*sizeof(int), cudaMemcpyDeviceToHost);
    run_kernel(kernel_num, input_gpu, output_gpu, w, h);
    cudaMemcpy(output, output_gpu, w*h*sizeof(int), cudaMemcpyDeviceToHost);
    for(int i=0; i<w*h; ++i){
        if(output[i] != output_ref[i]){
            printf("Vertification Failed!\n");
            break;
        }
    }
    cudaFree(input_gpu);
    cudaFree(output_gpu);
    delete[] input;
    delete[] output;
    delete[] output_ref;
    return 0;
}