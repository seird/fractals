#ifndef __COMPLEX_AVX_H__
#define __COMPLEX_AVX_H__


#ifdef __AVX2__

#include <immintrin.h>


void _mm256_autocmul_ps(__m256 * result_real, __m256 * result_imag,
                        __m256 * X_real, __m256 * X_imag);

void _mm256_cmul_ps(__m256 * result_real, __m256 * result_imag,
                    __m256 * X_real, __m256 * X_imag,
                    __m256 * Y_real, __m256 * Y_imag);

void _mm256_cdiv_ps(__m256 * result_real, __m256 * result_imag,
                    __m256 * X_real, __m256 * X_imag,
                    __m256 * Y_real, __m256 * Y_imag);

#endif // __AVX2__                  

#endif
