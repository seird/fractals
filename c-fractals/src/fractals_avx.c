#include "fractals_avx.h"


// Regular fractals

void
fractal_avxf_z2(__m256 * result_real, __m256 * result_imag, 
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
fractal_avxf_zn(__m256 * result_real, __m256 * result_imag, 
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
fractal_avxf_z3(__m256 * result_real, __m256 * result_imag, 
                     __m256 * z_real, __m256 * z_imag, 
                     __m256 * c_real, __m256 * c_imag)
{
    fractal_avxf_zn(result_real, result_imag, z_real, z_imag, c_real, c_imag, 3);
}

void
fractal_avxf_z4(__m256 * result_real, __m256 * result_imag, 
                     __m256 * z_real, __m256 * z_imag, 
                     __m256 * c_real, __m256 * c_imag)
{
    fractal_avxf_zn(result_real, result_imag, z_real, z_imag, c_real, c_imag, 4);
}


// Conjugate fractals

void
fractal_avxf_zconj2(__m256 * result_real, __m256 * result_imag, 
                   __m256 * z_real, __m256 * z_imag, 
                   __m256 * c_real, __m256 * c_imag)
{
    // z * z = (a+bj)*(a+bj) = a*a - b*b + 2(a*b)j
    // conjugate
    *z_imag = _mm256_xor_ps(*z_imag, _mm256_set1_ps(-0.0f));

    _mm256_autocmul_ps(result_real, result_imag, z_real, z_imag);

    // z_real*z_real + c_real
    *result_real = _mm256_add_ps(*result_real, *c_real);

    // z_imag*z_imag + c_imag
    *result_imag = _mm256_add_ps(*result_imag, *c_imag);
}

void
fractal_avxf_zconjn(__m256 * result_real, __m256 * result_imag, 
                    __m256 * z_real, __m256 * z_imag, 
                    __m256 * c_real, __m256 * c_imag,
                    int n)
{
    // conjugate
    *z_imag = _mm256_xor_ps(*z_imag, _mm256_set1_ps(-0.0f));

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
fractal_avxf_zconj3(__m256 * result_real, __m256 * result_imag, 
                    __m256 * z_real, __m256 * z_imag, 
                    __m256 * c_real, __m256 * c_imag)
{
    fractal_avxf_zconjn(result_real, result_imag, z_real, z_imag, c_real, c_imag, 3);
}

void
fractal_avxf_zconj4(__m256 * result_real, __m256 * result_imag, 
                    __m256 * z_real, __m256 * z_imag, 
                    __m256 * c_real, __m256 * c_imag)
{
    fractal_avxf_zconjn(result_real, result_imag, z_real, z_imag, c_real, c_imag, 4);
}


// Absolute value fractals

void
fractal_avxf_zabs2(__m256 * result_real, __m256 * result_imag, 
                    __m256 * z_real, __m256 * z_imag, 
                    __m256 * c_real, __m256 * c_imag)
{
    // z * z = (a+bj)*(a+bj) = a*a - b*b + 2(a*b)j
    // abs
    __m256 sign_mask = _mm256_set1_ps(-0.0f);
    *z_imag = _mm256_andnot_ps(sign_mask, *z_imag);
    *z_real = _mm256_andnot_ps(sign_mask, *z_real);

    _mm256_autocmul_ps(result_real, result_imag, z_real, z_imag);

    // z_real*z_real + c_real
    *result_real = _mm256_add_ps(*result_real, *c_real);

    // z_imag*z_imag + c_imag
    *result_imag = _mm256_add_ps(*result_imag, *c_imag);
}

void
fractal_avxf_zabsn(__m256 * result_real, __m256 * result_imag, 
                    __m256 * z_real, __m256 * z_imag, 
                    __m256 * c_real, __m256 * c_imag,
                    int n)
{
    // abs
    __m256 sign_mask = _mm256_set1_ps(-0.0f);
    *z_imag = _mm256_andnot_ps(sign_mask, *z_imag);
    *z_real = _mm256_andnot_ps(sign_mask, *z_real);

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
fractal_avxf_zabs3(__m256 * result_real, __m256 * result_imag, 
                    __m256 * z_real, __m256 * z_imag, 
                    __m256 * c_real, __m256 * c_imag)
{
    fractal_avxf_zabsn(result_real, result_imag, z_real, z_imag, c_real, c_imag, 3);
}

void
fractal_avxf_zabs4(__m256 * result_real, __m256 * result_imag, 
                    __m256 * z_real, __m256 * z_imag, 
                    __m256 * c_real, __m256 * c_imag)
{
    fractal_avxf_zabsn(result_real, result_imag, z_real, z_imag, c_real, c_imag, 4);
}


// Magnet fractals

void
fractal_avxf_magnet(__m256 * result_real, __m256 * result_imag, 
                    __m256 * z_real, __m256 * z_imag, 
                    __m256 * c_real, __m256 * c_imag)
{
    /* NUMERATOR */
    // z^2
    __m256 z_sq_real, z_sq_imag;
    _mm256_autocmul_ps(&z_sq_real, &z_sq_imag, z_real, z_imag);

    // z^2 + c - 1 = numerator
    __m256 numerator_real = _mm256_add_ps(
        _mm256_add_ps(z_sq_real, *c_real),
        _mm256_set1_ps(-1.0f)
    );
    __m256 numerator_imag = _mm256_add_ps(z_sq_imag, *c_imag);

    /* DENOMINATOR */
    // 2z
    __m256 two = _mm256_set1_ps(2.0f);
    __m256 _2z_real = _mm256_mul_ps(*z_real, two);
    __m256 _2z_imag = _mm256_mul_ps(*z_imag, two);

    // 2z + c - 2
    __m256 denominator_real = _mm256_add_ps(
        _mm256_add_ps(_2z_real, *c_real),
        _mm256_set1_ps(-2.0f)
    );
    __m256 denominator_imag = _mm256_add_ps(_2z_imag, *c_imag);

    /* FRACTION */
    // (a+bj)/(c+dj) = [(a+bj)*(c-dj)]/[(c+dj)*(c-dj)]
    __m256 fraction_real, fraction_imag;
    _mm256_cdiv_ps(&fraction_real, &fraction_imag,
                   &numerator_real, &numerator_imag,
                   &denominator_real, &denominator_imag);

    /* SQUARED FRACTION */
    _mm256_autocmul_ps(result_real, result_imag, &fraction_real, &fraction_imag);
}

// Julia variant z^2 + c/z

void
fractal_avxf_z2_z(__m256 * result_real, __m256 * result_imag, 
                  __m256 * z_real, __m256 * z_imag, 
                  __m256 * c_real, __m256 * c_imag)
{
    // z * z = (a+bj)*(a+bj) = a*a - b*b + 2(a*b)j
    _mm256_autocmul_ps(result_real, result_imag, z_real, z_imag);

    // c / z
    __m256 fraction_real, fraction_imag;
    _mm256_cdiv_ps(&fraction_real, &fraction_imag,
                   c_real, c_imag,
                   z_real, z_imag);

    // z_real*z_real + c_real/z
    *result_real = _mm256_add_ps(*result_real, fraction_real);

    // z_imag*z_imag + c_imag
    *result_imag = _mm256_add_ps(*result_imag, fraction_imag);
}


void (*fractal_avx_get(enum Fractal frac))(__m256 * result_real, __m256 * result_imag, 
                                        __m256 * z_real, __m256 * z_imag, 
                                        __m256 * c_real, __m256 * c_imag)
{
    void (*fptr)(__m256 * , __m256 *, __m256 *, __m256 *, __m256 *, __m256 *) = &fractal_avxf_z2;

    switch (frac) 
    {
        case FRAC_Z2: 
            fptr = &fractal_avxf_z2;
            break;
        case FRAC_Z3:
            fptr = &fractal_avxf_z3;
            break;
        case FRAC_Z4:
            fptr = &fractal_avxf_z4;
            break;
        case FRAC_ZCONJ2: 
            fptr = &fractal_avxf_zconj2;
            break;
        case FRAC_ZCONJ3:
            fptr = &fractal_avxf_zconj3;
            break;
        case FRAC_ZCONJ4:
            fptr = &fractal_avxf_zconj4;
            break;
        case FRAC_ZABS2: 
            fptr = &fractal_avxf_zabs2;
            break;
        case FRAC_ZABS3:
            fptr = &fractal_avxf_zabs3;
            break;
        case FRAC_ZABS4:
            fptr = &fractal_avxf_zabs4;
            break;
        case FRAC_MAGNET:
            fptr = &fractal_avxf_magnet;
            break;
        case FRAC_Z2_Z:
            fptr = &fractal_avxf_z2_z;
            break;
        default:
            fptr = &fractal_avxf_z2;
    }
    return fptr;
}
