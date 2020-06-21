#include "complex_avx.h"


void
_mm256_autocmul_ps(__m256 * result_real, __m256 * result_imag,
                   __m256 * X_real, __m256 * X_imag)
{
    // multiply a complex number with itself
    // x = a + bj
    //
    // x * x = (a+bj)*(a+bj) = a*a - b*b + 2(a*b)j

    // X_real*X_real
    __m256 X_real_sq = _mm256_mul_ps(*X_real, *X_real);

    // X_imag*X_imag
    __m256 X_imag_sq = _mm256_mul_ps(*X_imag, *X_imag);

    // 2*X_real*X_imag
    *result_imag = _mm256_mul_ps(
        _mm256_mul_ps(*X_real, *X_imag),
        _mm256_set1_ps(2)
    );

    // X_real*X_real - X_imag*X_imag
    *result_real = _mm256_sub_ps(X_real_sq, X_imag_sq);
}

void
_mm256_cmul_ps(__m256 * result_real, __m256 * result_imag,
               __m256 * X_real, __m256 * X_imag,
               __m256 * Y_real, __m256 * Y_imag)
{
    // multiply two complex numbers
    // x = a + bj
    // y = c + dj
    //
    // x * y = (a+bj)*(c+dj) = a*c - b*d + [a*d + b*c]j

    // X_real*Y_real
    __m256 AC = _mm256_mul_ps(*X_real, *Y_real);

    // X_imag*Y_imag
    __m256 BD = _mm256_mul_ps(*X_imag, *Y_imag);

    // X_real*Y_imag
    __m256 AD = _mm256_mul_ps(*X_real, *Y_imag);

    // X_imag*Y_real
    __m256 BC = _mm256_mul_ps(*X_imag, *Y_real);

    // X_real*X_real - X_imag*X_imag
    *result_real = _mm256_sub_ps(AC, BD);

    // X_real*Y_imag + X_imag*Y_real
    *result_imag = _mm256_add_ps(AD, BC);
}

void
_mm256_cdiv_ps(__m256 * result_real, __m256 * result_imag,
               __m256 * X_real, __m256 * X_imag,
               __m256 * Y_real, __m256 * Y_imag)
{
    // divide two complex numbers
    // x / y

    __m256 Y_imag_conj = _mm256_xor_ps(*Y_imag, _mm256_set1_ps(-0.0f));
    // [(a+bj)*(c-dj)]
    __m256 temp_num_real, temp_num_imag;
    _mm256_cmul_ps(&temp_num_real, &temp_num_imag,
                   X_real, X_imag,
                   Y_real, &Y_imag_conj);

    // [(c+dj)*(c-dj)] --> imag result is 0
    __m256 temp_den_real, temp_den_imag;
    _mm256_cmul_ps(&temp_den_real, &temp_den_imag,
                   Y_real, Y_imag,
                   Y_real, &Y_imag_conj);

    // [(a+bj)*(c-dj)]/[(c+dj)*(c-dj)]
    *result_real = _mm256_div_ps(temp_num_real, temp_den_real);
    *result_imag = _mm256_div_ps(temp_num_imag, temp_den_real);
}
