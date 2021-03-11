#pragma once


extern int * d_image;


#include <math.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include "../include/fractal_cuda.h"


#define BLACK 0


#ifdef __linux__
extern "C" bool fractal_cuda_init(int width, int height);
extern "C" void fractal_cuda_clean();
extern "C" void fractal_cuda_get_colors(int * image, struct FractalProperties * fp);
#else
extern "C" bool __declspec(dllexport) fractal_cuda_init(int width, int height);
extern "C" void __declspec(dllexport) fractal_cuda_clean();
extern "C" void __declspec(dllexport) fractal_cuda_get_colors(int * image, struct FractalProperties * fp);
#endif


typedef void (* fractal_cuda_t)(float * result_real, float * result_imag,
                                const float z_real, const float z_imag,
                                const float c_real, const float c_imag);


typedef void (* fractal_cuda_kernel_t)(int * colors, const float w_start, const float w_end,
                                       const float h_start, const float h_end,
                                       const float c_real, const float c_imag,
                                       int width, int height,
                                       int max_iterations, float R);


__device__ void fractal_cuda_z2(float * result_real, float * result_imag,
                                const float z_real, const float z_imag,
                                const float c_real, const float c_imag);

__device__ void fractal_cuda_z3(float * result_real, float * result_imag,
                                const float z_real, const float z_imag,
                                const float c_real, const float c_imag);
                                
__device__ void fractal_cuda_z4(float * result_real, float * result_imag,
                                const float z_real, const float z_imag,
                                const float c_real, const float c_imag);

__device__ void fractal_cuda_zconj2(float * result_real, float * result_imag,
                                    const float z_real, const float z_imag,
                                    const float c_real, const float c_imag);

__device__ void fractal_cuda_zconj3(float * result_real, float * result_imag,
                                    const float z_real, const float z_imag,
                                    const float c_real, const float c_imag);
                                
__device__ void fractal_cuda_zconj4(float * result_real, float * result_imag,
                                    const float z_real, const float z_imag,
                                    const float c_real, const float c_imag);

__device__ void fractal_cuda_zabs2(float * result_real, float * result_imag,
                                   const float z_real, const float z_imag,
                                   const float c_real, const float c_imag);

__device__ void fractal_cuda_zabs3(float * result_real, float * result_imag,
                                   const float z_real, const float z_imag,
                                   const float c_real, const float c_imag);
                                
__device__ void fractal_cuda_zabs4(float * result_real, float * result_imag,
                                   const float z_real, const float z_imag,
                                   const float c_real, const float c_imag);

__device__ void fractal_cuda_magnet(float * result_real, float * result_imag,
                                    const float z_real, const float z_imag,
                                    const float c_real, const float c_imag);

__device__ void fractal_cuda_z2_z(float * result_real, float * result_imag,
                                  const float z_real, const float z_imag,
                                  const float c_real, const float c_imag);



__device__ bool fractal_cuda_escape_magnitude_check(float z_real, float z_imag, float R);


__device__ void fractal_cuda_kernel_julia(int * colors, const float w_start, const float w_end,
                                                    const float h_start, const float h_end,
                                                    const float c_real, const float c_imag,
                                                    int width, int height,
                                                    int max_iterations, float R,
                                                    fractal_cuda_t fractal);

__global__ void fractal_cuda_kernel_julia_z2(int * colors, const float w_start, const float w_end,
                                                       const float h_start, const float h_end,
                                                       const float c_real, const float c_imag,
                                                       int width, int height,
                                                       int max_iterations, float R);

__global__ void fractal_cuda_kernel_julia_z3(int * colors, const float w_start, const float w_end,
                                                       const float h_start, const float h_end,
                                                       const float c_real, const float c_imag,
                                                       int width, int height,
                                                       int max_iterations, float R);

__global__ void fractal_cuda_kernel_julia_z4(int * colors, const float w_start, const float w_end,
                                                       const float h_start, const float h_end,
                                                       const float c_real, const float c_imag,
                                                       int width, int height,
                                                       int max_iterations, float R);

__global__ void fractal_cuda_kernel_julia_zconj2(int * colors, const float w_start, const float w_end,
                                                           const float h_start, const float h_end,
                                                           const float c_real, const float c_imag,
                                                           int width, int height,
                                                           int max_iterations, float R);

__global__ void fractal_cuda_kernel_julia_zconj3(int * colors, const float w_start, const float w_end,
                                                           const float h_start, const float h_end,
                                                           const float c_real, const float c_imag,
                                                           int width, int height,
                                                           int max_iterations, float R);

__global__ void fractal_cuda_kernel_julia_zconj4(int * colors, const float w_start, const float w_end,
                                                           const float h_start, const float h_end,
                                                           const float c_real, const float c_imag,
                                                           int width, int height,
                                                           int max_iterations, float R);

__global__ void fractal_cuda_kernel_julia_zabs2(int * colors, const float w_start, const float w_end,
                                                          const float h_start, const float h_end,
                                                          const float c_real, const float c_imag,
                                                          int width, int height,
                                                          int max_iterations, float R);

__global__ void fractal_cuda_kernel_julia_zabs3(int * colors, const float w_start, const float w_end,
                                                          const float h_start, const float h_end,
                                                          const float c_real, const float c_imag,
                                                          int width, int height,
                                                          int max_iterations, float R);

__global__ void fractal_cuda_kernel_julia_zabs4(int * colors, const float w_start, const float w_end,
                                                          const float h_start, const float h_end,
                                                          const float c_real, const float c_imag,
                                                          int width, int height,
                                                          int max_iterations, float R);

__global__ void fractal_cuda_kernel_julia_magnet(int * colors, const float w_start, const float w_end,
                                                           const float h_start, const float h_end,
                                                           const float c_real, const float c_imag,
                                                           int width, int height,
                                                           int max_iterations, float R);

__global__ void fractal_cuda_kernel_julia_z2_z(int * colors, const float w_start, const float w_end,
                                                         const float h_start, const float h_end,
                                                         const float c_real, const float c_imag,
                                                         int width, int height,
                                                         int max_iterations, float R);


__device__ void fractal_cuda_kernel_mandelbrot(int * colors, const float w_start, const float w_end,
                                                    const float h_start, const float h_end,
                                                    const float c_real, const float c_imag,
                                                    int width, int height,
                                                    int max_iterations, float R,
                                                    fractal_cuda_t fractal);

__global__ void fractal_cuda_kernel_mandelbrot_z2(int * colors, const float w_start, const float w_end,
                                                       const float h_start, const float h_end,
                                                       const float c_real, const float c_imag,
                                                       int width, int height,
                                                       int max_iterations, float R);

__global__ void fractal_cuda_kernel_mandelbrot_z3(int * colors, const float w_start, const float w_end,
                                                       const float h_start, const float h_end,
                                                       const float c_real, const float c_imag,
                                                       int width, int height,
                                                       int max_iterations, float R);

__global__ void fractal_cuda_kernel_mandelbrot_z4(int * colors, const float w_start, const float w_end,
                                                       const float h_start, const float h_end,
                                                       const float c_real, const float c_imag,
                                                       int width, int height,
                                                       int max_iterations, float R);

__global__ void fractal_cuda_kernel_mandelbrot_zconj2(int * colors, const float w_start, const float w_end,
                                                           const float h_start, const float h_end,
                                                           const float c_real, const float c_imag,
                                                           int width, int height,
                                                           int max_iterations, float R);

__global__ void fractal_cuda_kernel_mandelbrot_zconj3(int * colors, const float w_start, const float w_end,
                                                           const float h_start, const float h_end,
                                                           const float c_real, const float c_imag,
                                                           int width, int height,
                                                           int max_iterations, float R);

__global__ void fractal_cuda_kernel_mandelbrot_zconj4(int * colors, const float w_start, const float w_end,
                                                           const float h_start, const float h_end,
                                                           const float c_real, const float c_imag,
                                                           int width, int height,
                                                           int max_iterations, float R);

__global__ void fractal_cuda_kernel_mandelbrot_zabs2(int * colors, const float w_start, const float w_end,
                                                          const float h_start, const float h_end,
                                                          const float c_real, const float c_imag,
                                                          int width, int height,
                                                          int max_iterations, float R);

__global__ void fractal_cuda_kernel_mandelbrot_zabs3(int * colors, const float w_start, const float w_end,
                                                          const float h_start, const float h_end,
                                                          const float c_real, const float c_imag,
                                                          int width, int height,
                                                          int max_iterations, float R);

__global__ void fractal_cuda_kernel_mandelbrot_zabs4(int * colors, const float w_start, const float w_end,
                                                          const float h_start, const float h_end,
                                                          const float c_real, const float c_imag,
                                                          int width, int height,
                                                          int max_iterations, float R);

__global__ void fractal_cuda_kernel_mandelbrot_magnet(int * colors, const float w_start, const float w_end,
                                                           const float h_start, const float h_end,
                                                           const float c_real, const float c_imag,
                                                           int width, int height,
                                                           int max_iterations, float R);

__global__ void fractal_cuda_kernel_mandelbrot_z2_z(int * colors, const float w_start, const float w_end,
                                                         const float h_start, const float h_end,
                                                         const float c_real, const float c_imag,
                                                         int width, int height,
                                                         int max_iterations, float R);
