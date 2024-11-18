#include <cublas_v2.h>
#include <cuda_runtime.h>

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
);