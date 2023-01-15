#ifndef __FRACTALS_AVX_H__
#define __FRACTALS_AVX_H__


#if defined(__AVX2__) || defined(__AVX512DQ__)

#include "complex_avx.h"
#include "../include/fractal_color.h"
#include <immintrin.h>

#endif // __AVX2__ || __AVX512DQ__


#ifdef __AVX2__

typedef void (* fractal_avx_t)(__m256 * result_real, __m256 * result_imag, 
                            __m256 * z_real, __m256 * z_imag, 
                            __m256 * c_real, __m256 * c_imag);

typedef void (* fractal_avx_n_t)(__m256 * result_real, __m256 * result_imag, 
                             __m256 * z_real, __m256 * z_imag, 
                             __m256 * c_real, __m256 * c_imag);


// Regular fractals

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


// Conjugate fractals

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


// Absolute value fractals

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


// Magnet fractals

void fractal_avxf_magnet(__m256 * result_real, __m256 * result_imag, 
                         __m256 * z_real, __m256 * z_imag, 
                         __m256 * c_real, __m256 * c_imag);

// Julia variant

void fractal_avxf_z2_z(__m256 * result_real, __m256 * result_imag, 
                       __m256 * z_real, __m256 * z_imag, 
                       __m256 * c_real, __m256 * c_imag);



fractal_avx_t fractal_avx_get(enum FC_Fractal frac);

#endif // __AVX2__


#ifdef __AVX512DQ__

typedef void (* fractal_avx512_t)(__m512 * result_real, __m512 * result_imag, 
                            __m512 * z_real, __m512 * z_imag, 
                            __m512 * c_real, __m512 * c_imag);

typedef void (* fractal_avx512_n_t)(__m512 * result_real, __m512 * result_imag, 
                             __m512 * z_real, __m512 * z_imag, 
                             __m512 * c_real, __m512 * c_imag);


// Regular fractals

void fractal_avx512f_z2(__m512 * result_real, __m512 * result_imag, 
                        __m512 * z_real, __m512 * z_imag, 
                        __m512 * c_real, __m512 * c_imag);

void fractal_avx512f_zn(__m512 * result_real, __m512 * result_imag, 
                          __m512 * z_real, __m512 * z_imag, 
                          __m512 * c_real, __m512 * c_imag,
                          int n);

void fractal_avx512f_z3(__m512 * result_real, __m512 * result_imag, 
                          __m512 * z_real, __m512 * z_imag, 
                          __m512 * c_real, __m512 * c_imag);

void fractal_avx512f_z4(__m512 * result_real, __m512 * result_imag, 
                          __m512 * z_real, __m512 * z_imag, 
                          __m512 * c_real, __m512 * c_imag);


// Conjugate fractals

void fractal_avx512f_zconj2(__m512 * result_real, __m512 * result_imag, 
                        __m512 * z_real, __m512 * z_imag, 
                        __m512 * c_real, __m512 * c_imag);

void fractal_avx512f_zconjn(__m512 * result_real, __m512 * result_imag, 
                         __m512 * z_real, __m512 * z_imag, 
                         __m512 * c_real, __m512 * c_imag,
                         int n);

void fractal_avx512f_zconj3(__m512 * result_real, __m512 * result_imag, 
                         __m512 * z_real, __m512 * z_imag, 
                         __m512 * c_real, __m512 * c_imag);

void fractal_avx512f_zconj4(__m512 * result_real, __m512 * result_imag, 
                         __m512 * z_real, __m512 * z_imag, 
                         __m512 * c_real, __m512 * c_imag);


// Absolute value fractals

void fractal_avx512f_zabs2(__m512 * result_real, __m512 * result_imag, 
                        __m512 * z_real, __m512 * z_imag, 
                        __m512 * c_real, __m512 * c_imag);

void fractal_avx512f_zabsn(__m512 * result_real, __m512 * result_imag, 
                         __m512 * z_real, __m512 * z_imag, 
                         __m512 * c_real, __m512 * c_imag,
                         int n);

void fractal_avx512f_zabs3(__m512 * result_real, __m512 * result_imag, 
                         __m512 * z_real, __m512 * z_imag, 
                         __m512 * c_real, __m512 * c_imag);

void fractal_avx512f_zabs4(__m512 * result_real, __m512 * result_imag, 
                         __m512 * z_real, __m512 * z_imag, 
                         __m512 * c_real, __m512 * c_imag);


// Magnet fractals

void fractal_avx512f_magnet(__m512 * result_real, __m512 * result_imag, 
                         __m512 * z_real, __m512 * z_imag, 
                         __m512 * c_real, __m512 * c_imag);

// Julia variant

void fractal_avx512f_z2_z(__m512 * result_real, __m512 * result_imag, 
                       __m512 * z_real, __m512 * z_imag, 
                       __m512 * c_real, __m512 * c_imag);



fractal_avx512_t fractal_avx512_get(enum FC_Fractal frac);

#endif // __AVX512DQ__

#endif
