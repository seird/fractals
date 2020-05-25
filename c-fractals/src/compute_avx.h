#ifndef __COMPUTE_AVX_H__
#define __COMPUTE_AVX_H__


#include <stdio.h>
#include <immintrin.h>

#include "fractal_color.h"
#include "fractals_avx.h"
#include "main.h"


#define AVX_ALIGNMENT 32
#define VECFSIZE 8


void fractal_avxf_print_vec(__m256 * v);

void fractal_avxf_escape_magnitude_check(__m256 * escaped_mask, 
                                         __m256 * z_real, __m256 * z_imag,
                                         __m256 * RR);

void fractal_avxf_get_vector_color(FRACDTYPE * color_array, 
                                   __m256 * z_real, __m256 * z_imag,
                                   __m256 * c_real, __m256 * c_imag,
                                   __m256 * RR, int max_iterations,
                                   void (*fractal)(__m256 *, __m256 *, __m256 *, __m256 *, __m256 * , __m256 *));

#endif
