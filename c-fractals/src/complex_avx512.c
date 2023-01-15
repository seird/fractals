#include "complex_avx.h"


#ifdef __AVX512DQ__

void
_mm512_autocmul_ps(__m512 * result_real, __m512 * result_imag,
                   __m512 * X_real, __m512 * X_imag)
{
    // multiply a complex number with itself
    // x = a + bj
    //
    // x * x = (a+bj)*(a+bj) = a*a - b*b + 2(a*b)j

    // X_real*X_real
    __m512 X_real_sq = _mm512_mul_ps(*X_real, *X_real);

    // X_imag*X_imag
    __m512 X_imag_sq = _mm512_mul_ps(*X_imag, *X_imag);

    // 2*X_real*X_imag
    *result_imag = _mm512_mul_ps(
        _mm512_mul_ps(*X_real, *X_imag),
        _mm512_set1_ps(2)
    );

    // X_real*X_real - X_imag*X_imag
    *result_real = _mm512_sub_ps(X_real_sq, X_imag_sq);
}

void
_mm512_cmul_ps(__m512 * result_real, __m512 * result_imag,
               __m512 * X_real, __m512 * X_imag,
               __m512 * Y_real, __m512 * Y_imag)
{
    // multiply two complex numbers
    // x = a + bj
    // y = c + dj
    //
    // x * y = (a+bj)*(c+dj) = a*c - b*d + [a*d + b*c]j

    // X_real*Y_real
    __m512 AC = _mm512_mul_ps(*X_real, *Y_real);

    // X_imag*Y_imag
    __m512 BD = _mm512_mul_ps(*X_imag, *Y_imag);

    // X_real*Y_imag
    __m512 AD = _mm512_mul_ps(*X_real, *Y_imag);

    // X_imag*Y_real
    __m512 BC = _mm512_mul_ps(*X_imag, *Y_real);

    // X_real*X_real - X_imag*X_imag
    *result_real = _mm512_sub_ps(AC, BD);

    // X_real*Y_imag + X_imag*Y_real
    *result_imag = _mm512_add_ps(AD, BC);
}

void
_mm512_cdiv_ps(__m512 * result_real, __m512 * result_imag,
               __m512 * X_real, __m512 * X_imag,
               __m512 * Y_real, __m512 * Y_imag)
{
    // divide two complex numbers
    // x / y

    __m512 Y_imag_conj = _mm512_xor_ps(*Y_imag, _mm512_set1_ps(-0.0f));
    // [(a+bj)*(c-dj)]
    __m512 temp_num_real, temp_num_imag;
    _mm512_cmul_ps(&temp_num_real, &temp_num_imag,
                   X_real, X_imag,
                   Y_real, &Y_imag_conj);

    // [(c+dj)*(c-dj)] --> imag result is 0
    __m512 temp_den_real, temp_den_imag;
    _mm512_cmul_ps(&temp_den_real, &temp_den_imag,
                   Y_real, Y_imag,
                   Y_real, &Y_imag_conj);

    // [(a+bj)*(c-dj)]/[(c+dj)*(c-dj)]
    *result_real = _mm512_div_ps(temp_num_real, temp_den_real);
    *result_imag = _mm512_div_ps(temp_num_imag, temp_den_real);
}

#endif // __AVX512DQ__
