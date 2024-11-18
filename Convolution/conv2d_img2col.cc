#include "conv2d_img2col.h"


void img2col(
    const float* input,
    float* new_input,
    int input_width,
    int input_height,
    int kernel_size,
    int output_width,
    int output_height,
    int stride

){
    // input: [input_width, input_height]
    // output: [output_width*output_height,kernel_size*kernel_size]
    int ksize = kernel_size * kernel_size;
    for(int ow=0; ow<output_width; ++ow){
        for(int oh=0; oh<output_height; ++oh){
            for(int kw=0; kw<kernel_size; ++kw){
                for(int kh=0; kh<kernel_size; ++kh){
                    int iw = ow*stride+kw;
                    int ih = oh*stride+kh;
                    if(iw>=0&&iw<input_width&&ih>=0&&ih<input_height){
                        new_input[(oh*output_width+ow)*ksize+kh*kernel_size+kw] = input[ih*input_width+iw];
                    }
                }
            }
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
    float* new_input = new float[output_size, kernel_size*kernel_size];
    img2col(input, new_input, input_width, input_height, kernel_size, output_width, output_height, stride);
    cublasHandle_t handle;
    cublasCreate(&handle);
    cublasgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, output_size, 1, ksize, 1.0f, new_input, ksize, kernel, 1, 0.0f, output, 1);
    delete[] new_input;
    cublasDestroy(handle);
}