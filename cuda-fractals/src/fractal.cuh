#pragma once


#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include "../include/fractal_cuda.h"


extern uint8_t * d_image;


#define BLACK 0


#ifdef __linux__
extern "C" bool fractal_cuda_init(int width, int height);
extern "C" void fractal_cuda_clean();
extern "C" void fractal_cuda_get_colors(uint8_t * image, struct FractalProperties * fp);
#else
extern "C" bool __declspec(dllexport) fractal_cuda_init(int width, int height);
extern "C" void __declspec(dllexport) fractal_cuda_clean();
extern "C" void __declspec(dllexport) fractal_cuda_get_colors(uint8_t * image, struct FractalProperties * fp);
#endif


typedef void (* fractal_cuda_t)(float * result_real, float * result_imag,
                                const float z_real, const float z_imag,
                                const float c_real, const float c_imag);


typedef void (* fractal_cuda_kernel_t)(uint8_t * colors, const float w_start, const float w_end,
                                       const float h_start, const float h_end,
                                       const float c_real, const float c_imag,
                                       int width, int height,
                                       int max_iterations, float R,
                                       FC_Color color);
