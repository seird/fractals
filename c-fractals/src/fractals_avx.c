#include "fractals_avx.h"


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
    // multiply two complex number
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
fractal_avxf_julia(__m256 * result_real, __m256 * result_imag, 
                   __m256 * z_real, __m256 * z_imag, 
                   __m256 * c_real, __m256 * c_imag)
{
    // z * z = (a+bj)*(a+bj) = a*a - b*b + 2(a*b)j
    _mm256_autocmul_ps(result_real, result_imag, z_real, z_imag);

    // z_real*z_real + c_real
    *result_real = _mm256_add_ps(*result_real, *c_real);

    // z_imag*z_imag + c_imag
    *result_imag = _mm256_add_ps(*result_imag, *c_imag);
}

void
fractal_avxf_julia_n(__m256 * result_real, __m256 * result_imag, 
                     __m256 * z_real, __m256 * z_imag, 
                     __m256 * c_real, __m256 * c_imag,
                     int n)
{
    __m256 r_real = *z_real;
    __m256 r_imag = *z_imag;
    for (int i=1; i<n; ++i) {
        _mm256_cmul_ps(&r_real, &r_imag,
                       &r_real, &r_imag,
                       z_real, z_imag);
    }

    

    // z_real*z_real + c_real
    *result_real = _mm256_add_ps(r_real, *c_real);

    // z_imag*z_imag + c_imag
    *result_imag = _mm256_add_ps(r_imag, *c_imag);
}

void
fractal_avxf_julia_3(__m256 * result_real, __m256 * result_imag, 
                     __m256 * z_real, __m256 * z_imag, 
                     __m256 * c_real, __m256 * c_imag)
{
    fractal_avxf_julia_n(result_real, result_imag, z_real, z_imag, c_real, c_imag, 3);
}

void
fractal_avxf_julia_4(__m256 * result_real, __m256 * result_imag, 
                     __m256 * z_real, __m256 * z_imag, 
                     __m256 * c_real, __m256 * c_imag)
{
    fractal_avxf_julia_n(result_real, result_imag, z_real, z_imag, c_real, c_imag, 4);
}

void (*fractal_avx_get(enum Fractal frac))(__m256 * result_real, __m256 * result_imag, 
                                        __m256 * z_real, __m256 * z_imag, 
                                        __m256 * c_real, __m256 * c_imag)
{
    void (*fptr)(__m256 * , __m256 *, __m256 *, __m256 *, __m256 *, __m256 *) = &fractal_avxf_julia;

    switch (frac) 
    {
        case FRAC_Z2: 
            fptr = &fractal_avxf_julia;
            break;
        case FRAC_Z3:
            fptr = &fractal_avxf_julia_3;
            break;
        case FRAC_Z4:
            fptr = &fractal_avxf_julia_4;
            break;
        default:
            fptr = &fractal_avxf_julia;
    }
    return fptr;
}
