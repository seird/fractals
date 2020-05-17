#ifndef __COMPUTE_AVX_H__
#define __COMPUTE_AVX_H__


#include <stdio.h>
#include <immintrin.h>


#define AVX_ALIGNMENT 32
#define VECFSIZE 8


void fractal_avxf_print_vec(__m256 * v);
void fractal_avxf_julia(__m256 * result_real, __m256 * result_imag, 
                        __m256 * z_real, __m256 * z_imag,
                        __m256 * c_real, __m256 * c_imag);


#endif
