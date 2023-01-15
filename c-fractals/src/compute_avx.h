#ifndef __COMPUTE_AVX_H__
#define __COMPUTE_AVX_H__


#if defined(__AVX2__) || defined(__AVX512DQ__)

#include <stdio.h>
#include <immintrin.h>

#include "../include/fractal_color.h"
#include "fractals_avx.h"
#include "main.h"


#define AVX_ALIGNMENT 32
#define VECFSIZE 8

#define AVX512_ALIGNMENT 64
#define VEC512FSIZE 16

#endif // __AVX2__ || __AVX512DQ__


#ifdef __AVX2__

void fractal_avxf_print_vec(__m256 * v);

void fractal_avxf_escape_magnitude_check(__m256 * escaped_mask, 
                                         __m256 * z_real, __m256 * z_imag,
                                         __m256 * RR);

void fractal_avxf_get_vector_color(float * color_array, 
                                   __m256 * z_real, __m256 * z_imag,
                                   __m256 * c_real, __m256 * c_imag,
                                   __m256 * RR, int max_iterations,
                                   fractal_avx_t fractal);

#endif // __AVX2__


#ifdef __AVX512DQ__

__mmask16 fractal_avx512f_escape_magnitude_check(__m512 * z_real, __m512 * z_imag,
                                                 __m512 * RR);

void fractal_avx512f_get_vector_color(float * color_array, 
                                      __m512 * z_real, __m512 * z_imag,
                                      __m512 * c_real, __m512 * c_imag,
                                      __m512 * RR, int max_iterations,
                                      fractal_avx512_t fractal);


#endif // __AVX512DQ__


#endif //__COMPUTE_AVX_H__
