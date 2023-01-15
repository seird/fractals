#ifndef __COMPLEX_AVX_H__
#define __COMPLEX_AVX_H__


#if defined(__AVX2__) || defined(__AVX512DQ__)

#include <immintrin.h>

#endif // __AVX2__ || __AVX512DQ__


#ifdef __AVX2__

void _mm256_autocmul_ps(__m256 * result_real, __m256 * result_imag,
                        __m256 * X_real, __m256 * X_imag);

void _mm256_cmul_ps(__m256 * result_real, __m256 * result_imag,
                    __m256 * X_real, __m256 * X_imag,
                    __m256 * Y_real, __m256 * Y_imag);

void _mm256_cdiv_ps(__m256 * result_real, __m256 * result_imag,
                    __m256 * X_real, __m256 * X_imag,
                    __m256 * Y_real, __m256 * Y_imag);

#endif // __AVX2__    


#ifdef __AVX512DQ__

void _mm512_autocmul_ps(__m512 * result_real, __m512 * result_imag,
                        __m512 * X_real, __m512 * X_imag);

void _mm512_cmul_ps(__m512 * result_real, __m512 * result_imag,
                    __m512 * X_real, __m512 * X_imag,
                    __m512 * Y_real, __m512 * Y_imag);

void _mm512_cdiv_ps(__m512 * result_real, __m512 * result_imag,
                    __m512 * X_real, __m512 * X_imag,
                    __m512 * Y_real, __m512 * Y_imag);

#endif // __AVX512DQ__


#endif // __COMPLEX_AVX_H__
