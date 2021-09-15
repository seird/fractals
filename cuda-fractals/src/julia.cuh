#pragma once


#include "kernels.cuh"


__device__ void
fractal_cuda_kernel_julia(uint8_t * colors, const float w_start, const float w_end,
                                    const float h_start, const float h_end,
                                    const float c_real, const float c_imag,
                                    int width, int height,
                                    int max_iterations, float R,
                                    fractal_cuda_t fractal,
                                    FC_Color color)
{
    int w = (blockIdx.x*blockDim.x) + threadIdx.x;
    int h = (blockIdx.y*blockDim.y) + threadIdx.y;

    float z_real = w_start + (float)w/width * (w_end - w_start);
    float z_imag = h_start + (float)h/height * (h_end - h_start);

    int num_iterations = 0;
    float r_real;
    float r_imag;
    for (; num_iterations<max_iterations; ++num_iterations) {
        if (fractal_cuda_escape_magnitude_check(z_real, z_imag, R)) {
            break;
        }
        fractal(&r_real, &r_imag, z_real, z_imag, c_real, c_imag);
        z_real = r_real;
        z_imag = r_imag;
    }

    int value = num_iterations == max_iterations ? BLACK : num_iterations;

    colorfuncs[color%FC_COLOR_NUM_ENTRIES](
        colors + h*width*3 + w*3,
        colors + h*width*3 + w*3 + 1,
        colors + h*width*3 + w*3 + 2,
        value
    );
}


__global__ void
fractal_cuda_kernel_julia_z2(uint8_t * colors, const float w_start, const float w_end,
                                       const float h_start, const float h_end,
                                       const float c_real, const float c_imag,
                                       int width, int height,
                                       int max_iterations, float R,
                                       FC_Color color)
{
    fractal_cuda_kernel_julia(colors, w_start, w_end,
                                        h_start, h_end,
                                        c_real, c_imag,
                                        width, height,
                                        max_iterations, R,
                                        fractal_cuda_z2,
                                        color);
}

__global__ void
fractal_cuda_kernel_julia_z3(uint8_t * colors, const float w_start, const float w_end,
                                       const float h_start, const float h_end,
                                       const float c_real, const float c_imag,
                                       int width, int height,
                                       int max_iterations, float R,
                                       FC_Color color)
{
    fractal_cuda_kernel_julia(colors, w_start, w_end,
                                        h_start, h_end,
                                        c_real, c_imag,
                                        width, height,
                                        max_iterations, R,
                                        fractal_cuda_z3,
                                        color);
}

__global__ void
fractal_cuda_kernel_julia_z4(uint8_t * colors, const float w_start, const float w_end,
                                       const float h_start, const float h_end,
                                       const float c_real, const float c_imag,
                                       int width, int height,
                                       int max_iterations, float R,
                                       FC_Color color)
{
    fractal_cuda_kernel_julia(colors, w_start, w_end,
                                        h_start, h_end,
                                        c_real, c_imag,
                                        width, height,
                                        max_iterations, R,
                                        fractal_cuda_z4,
                                        color);
}

__global__ void
fractal_cuda_kernel_julia_zconj2(uint8_t * colors, const float w_start, const float w_end,
                                           const float h_start, const float h_end,
                                           const float c_real, const float c_imag,
                                           int width, int height,
                                           int max_iterations, float R,
                                           FC_Color color)
{
    fractal_cuda_kernel_julia(colors, w_start, w_end,
                                        h_start, h_end,
                                        c_real, c_imag,
                                        width, height,
                                        max_iterations, R,
                                        fractal_cuda_zconj2,
                                        color);
}

__global__ void
fractal_cuda_kernel_julia_zconj3(uint8_t * colors, const float w_start, const float w_end,
                                           const float h_start, const float h_end,
                                           const float c_real, const float c_imag,
                                           int width, int height,
                                           int max_iterations, float R,
                                           FC_Color color)
{
    fractal_cuda_kernel_julia(colors, w_start, w_end,
                                        h_start, h_end,
                                        c_real, c_imag,
                                        width, height,
                                        max_iterations, R,
                                        fractal_cuda_zconj3,
                                        color);
}

__global__ void
fractal_cuda_kernel_julia_zconj4(uint8_t * colors, const float w_start, const float w_end,
                                           const float h_start, const float h_end,
                                           const float c_real, const float c_imag,
                                           int width, int height,
                                           int max_iterations, float R,
                                           FC_Color color)
{
    fractal_cuda_kernel_julia(colors, w_start, w_end,
                                        h_start, h_end,
                                        c_real, c_imag,
                                        width, height,
                                        max_iterations, R,
                                        fractal_cuda_zconj4,
                                        color);
}

__global__ void
fractal_cuda_kernel_julia_zabs2(uint8_t * colors, const float w_start, const float w_end,
                                          const float h_start, const float h_end,
                                          const float c_real, const float c_imag,
                                          int width, int height,
                                          int max_iterations, float R,
                                          FC_Color color)
{
    fractal_cuda_kernel_julia(colors, w_start, w_end,
                                        h_start, h_end,
                                        c_real, c_imag,
                                        width, height,
                                        max_iterations, R,
                                        fractal_cuda_zabs2,
                                        color);
}

__global__ void
fractal_cuda_kernel_julia_zabs3(uint8_t * colors, const float w_start, const float w_end,
                                          const float h_start, const float h_end,
                                          const float c_real, const float c_imag,
                                          int width, int height,
                                          int max_iterations, float R,
                                          FC_Color color)
{
    fractal_cuda_kernel_julia(colors, w_start, w_end,
                                        h_start, h_end,
                                        c_real, c_imag,
                                        width, height,
                                        max_iterations, R,
                                        fractal_cuda_zabs3,
                                        color);
}

__global__ void
fractal_cuda_kernel_julia_zabs4(uint8_t * colors, const float w_start, const float w_end,
                                          const float h_start, const float h_end,
                                          const float c_real, const float c_imag,
                                          int width, int height,
                                          int max_iterations, float R,
                                          FC_Color color)
{
    fractal_cuda_kernel_julia(colors, w_start, w_end,
                                        h_start, h_end,
                                        c_real, c_imag,
                                        width, height,
                                        max_iterations, R,
                                        fractal_cuda_zabs4,
                                        color);
}

__global__ void
fractal_cuda_kernel_julia_magnet(uint8_t * colors, const float w_start, const float w_end,
                                           const float h_start, const float h_end,
                                           const float c_real, const float c_imag,
                                           int width, int height,
                                           int max_iterations, float R,
                                           FC_Color color)
{
    fractal_cuda_kernel_julia(colors, w_start, w_end,
                                        h_start, h_end,
                                        c_real, c_imag,
                                        width, height,
                                        max_iterations, R,
                                        fractal_cuda_magnet,
                                        color);
}

__global__ void
fractal_cuda_kernel_julia_z2_z(uint8_t * colors, const float w_start, const float w_end,
                                         const float h_start, const float h_end,
                                         const float c_real, const float c_imag,
                                         int width, int height,
                                         int max_iterations, float R,
                                         FC_Color color)
{
    fractal_cuda_kernel_julia(colors, w_start, w_end,
                                        h_start, h_end,
                                        c_real, c_imag,
                                        width, height,
                                        max_iterations, R,
                                        fractal_cuda_z2_z,
                                        color);
}
