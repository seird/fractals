#ifndef __COMPUTE_AVX_H__
#define __COMPUTE_AVX_H__


#include <stdio.h>
#include <immintrin.h>

#include "fractal_color.h"
#include "main.h"


#define AVX_ALIGNMENT 32
#define VECFSIZE 8


void fractal_avxf_print_vec(__m256 * v);

void fractal_avxf_julia(__m256 * result_real, __m256 * result_imag, 
                        __m256 * z_real, __m256 * z_imag,
                        __m256 * c_real, __m256 * c_imag);

void fractal_avxf_escape_magnitude_check(__m256 * escaped_mask, 
                                         __m256 * z_real, __m256 * z_imag,
                                         __m256 * RR);

void fractal_avxf_get_vector_color(FRACDTYPE * color_array, 
                                   __m256 * z_real, __m256 * z_imag,
                                   __m256 * c_real, __m256 * c_imag,
                                   __m256 * RR, int max_iterations);

#endif
