#include "conv2d_cpu.h"

void conv2d_cpu(
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
    for (int y = 0; y < output_height; ++y) {
        for (int x = 0; x < output_width; ++x) {
            float result = 0.0f;
            for (int ky = 0; ky < kernel_size; ++ky) {
                for (int kx = 0; kx < kernel_size; ++kx) {
                    int in_x = x * stride + kx;
                    int in_y = y * stride + ky;
                    if (in_x >= 0 && in_x < input_width && in_y >= 0 && in_y < input_height) {
                        result += input[in_y * input_width + in_x] * kernel[ky * kernel_size + kx];
                    }
                }
            }
            output[y * output_width + x] = result;
        }
    }
}


