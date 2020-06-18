#ifndef __FRACTALS_AVX_H__
#define __FRACTALS_AVX_H__


#include "fractal_color.h"
#include <immintrin.h>


void _mm256_autocmul_ps(__m256 * result_real, __m256 * result_imag,
                        __m256 * X_real, __m256 * X_imag);

void _mm256_cmul_ps(__m256 * result_real, __m256 * result_imag,
                    __m256 * X_real, __m256 * X_imag,
                    __m256 * Y_real, __m256 * Y_imag);


void fractal_avxf_z2(__m256 * result_real, __m256 * result_imag, 
                        __m256 * z_real, __m256 * z_imag, 
                        __m256 * c_real, __m256 * c_imag);

void fractal_avxf_zn(__m256 * result_real, __m256 * result_imag, 
                          __m256 * z_real, __m256 * z_imag, 
                          __m256 * c_real, __m256 * c_imag,
                          int n);

void fractal_avxf_z3(__m256 * result_real, __m256 * result_imag, 
                          __m256 * z_real, __m256 * z_imag, 
                          __m256 * c_real, __m256 * c_imag);

void fractal_avxf_z4(__m256 * result_real, __m256 * result_imag, 
                          __m256 * z_real, __m256 * z_imag, 
                          __m256 * c_real, __m256 * c_imag);


void fractal_avxf_zconj2(__m256 * result_real, __m256 * result_imag, 
                        __m256 * z_real, __m256 * z_imag, 
                        __m256 * c_real, __m256 * c_imag);

void fractal_avxf_zconjn(__m256 * result_real, __m256 * result_imag, 
                         __m256 * z_real, __m256 * z_imag, 
                         __m256 * c_real, __m256 * c_imag,
                         int n);

void fractal_avxf_zconj3(__m256 * result_real, __m256 * result_imag, 
                         __m256 * z_real, __m256 * z_imag, 
                         __m256 * c_real, __m256 * c_imag);

void fractal_avxf_zconj4(__m256 * result_real, __m256 * result_imag, 
                         __m256 * z_real, __m256 * z_imag, 
                         __m256 * c_real, __m256 * c_imag);


void fractal_avxf_zabs2(__m256 * result_real, __m256 * result_imag, 
                        __m256 * z_real, __m256 * z_imag, 
                        __m256 * c_real, __m256 * c_imag);

void fractal_avxf_zabsn(__m256 * result_real, __m256 * result_imag, 
                         __m256 * z_real, __m256 * z_imag, 
                         __m256 * c_real, __m256 * c_imag,
                         int n);

void fractal_avxf_zabs3(__m256 * result_real, __m256 * result_imag, 
                         __m256 * z_real, __m256 * z_imag, 
                         __m256 * c_real, __m256 * c_imag);

void fractal_avxf_zabs4(__m256 * result_real, __m256 * result_imag, 
                         __m256 * z_real, __m256 * z_imag, 
                         __m256 * c_real, __m256 * c_imag);

void (*fractal_avx_get(enum Fractal frac))(__m256 * result_real, __m256 * result_imag, 
                                           __m256 * z_real, __m256 * z_imag, 
                                           __m256 * c_real, __m256 * c_imag);

#endif
