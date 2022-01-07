#pragma once


#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include "../include/fractal_cuda.h"

#include "gradients.cuh"
#include "fractals.cuh"


extern uint8_t * d_image;


#define BLACK 0


typedef void (* fractal_cuda_t)(float * result_real, float * result_imag,
                                const float z_real, const float z_imag,
                                const float c_real, const float c_imag);


typedef void (* fractal_cuda_kernel_t)(uint8_t * colors, int width, int height, FractalProperties fp);

__device__ inline bool
fractal_cuda_escape_magnitude_check(float z_real, float z_imag, float R)
{
    return (z_real*z_real + z_imag*z_imag) > (R*R);
}
