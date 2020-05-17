#include "compute_avx.h"



void
fractal_avx_print_vec(__m256 * v)
{
    for (float * f = (float*)v; f < (float*)v + VECSIZE; ++f) {
        printf("%5f ", *f);
    }
    putchar('\n');
}

void
fractal_avx_julia(__m256 * result_real, __m256 * result_imag, __m256 * z_real, __m256 * z_imag, __m256 * c_real, __m256 * c_imag)
{
    // z * z = (a+bj)*(a+bj) = a*a - b*b + 2(a*b)j

    // z_real*z_real
    __m256 r1 = _mm256_mul_ps(*z_real, *z_real);

    // z_imag*z_imag
    __m256 r2 = _mm256_mul_ps(*z_imag, *z_imag);

    // 2*z_real*z_imag
    __m256 r3 = _mm256_mul_ps(
        _mm256_mul_ps(*z_real, *z_imag),
        _mm256_set1_ps(2)
    );

    // z_real*z_real - z_imag*z_imag
    __m256 r4 = _mm256_sub_ps(r1, r2);

    // real(z*z) + c_real
    *result_real = _mm256_add_ps(r4, *c_real);

    // imag(z*z) + c_imag
    *result_imag = _mm256_add_ps(r3, *c_imag);
}
