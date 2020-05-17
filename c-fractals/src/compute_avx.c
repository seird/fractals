#include "compute_avx.h"



void
fractal_avxf_print_vec(__m256 * v)
{
    for (float * f = (float*)v; f < (float*)v + VECFSIZE; ++f) {
        printf("%5f ", *f);
    }
    putchar('\n');
}

void
fractal_avxf_julia(__m256 * result_real, __m256 * result_imag, 
                   __m256 * z_real, __m256 * z_imag, 
                   __m256 * c_real, __m256 * c_imag)
{
    // z * z = (a+bj)*(a+bj) = a*a - b*b + 2(a*b)j

    // z_real*z_real
    __m256 z_real_sq = _mm256_mul_ps(*z_real, *z_real);

    // z_imag*z_imag
    __m256 z_imag_sq = _mm256_mul_ps(*z_imag, *z_imag);

    // 2*z_real*z_imag
    __m256 z_real_imag_prod = _mm256_mul_ps(
        _mm256_mul_ps(*z_real, *z_imag),
        _mm256_set1_ps(2)
    );

    // z_real*z_real - z_imag*z_imag
    __m256 z_sq_diff = _mm256_sub_ps(z_real_sq, z_imag_sq);

    // real(z*z) + c_real
    *result_real = _mm256_add_ps(z_sq_diff, *c_real);

    // imag(z*z) + c_imag
    *result_imag = _mm256_add_ps(z_real_imag_prod, *c_imag);
}

void
fractal_avx_escape_magnitude_check(__m256 * escaped_mask,
                                   __m256 * z_real, __m256 * z_imag,
                                   __m256 * RR)
{
    // z_real * z_real + z_imag * z_imag > R*R

    // z_real * z_real
    __m256 z_real_sq = _mm256_mul_ps(*z_real, *z_real);

    // z_imag * z_imag
    __m256 z_imag_sq = _mm256_mul_ps(*z_imag, *z_imag);

    // z_real * z_real + z_imag * z_imag
    __m256 z_sq_sum = _mm256_add_ps(z_real_sq, z_imag_sq);

    // z_real * z_real + z_imag * z_imag > R*R
    *escaped_mask = _mm256_cmp_ps(z_sq_sum, *RR, _CMP_GT_OQ); // dst[i+31:i] := ( a[i+31:i] OP b[i+31:i] ) ? 0xFFFFFFFF : 0
}
