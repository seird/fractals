#include "fractals_avx.h"

#ifdef __AVX512DQ__

// Regular fractals

void
fractal_avx512f_z2(__m512 * result_real, __m512 * result_imag, 
                   __m512 * z_real, __m512 * z_imag, 
                   __m512 * c_real, __m512 * c_imag)
{
    // z * z = (a+bj)*(a+bj) = a*a - b*b + 2(a*b)j
    _mm512_autocmul_ps(result_real, result_imag, z_real, z_imag);

    // z_real*z_real + c_real
    *result_real = _mm512_add_ps(*result_real, *c_real);

    // z_imag*z_imag + c_imag
    *result_imag = _mm512_add_ps(*result_imag, *c_imag);
}

void
fractal_avx512f_zn(__m512 * result_real, __m512 * result_imag, 
                     __m512 * z_real, __m512 * z_imag, 
                     __m512 * c_real, __m512 * c_imag,
                     int n)
{
    __m512 r_real = *z_real;
    __m512 r_imag = *z_imag;
    for (int i=1; i<n; ++i) {
        _mm512_cmul_ps(&r_real, &r_imag,
                       &r_real, &r_imag,
                       z_real, z_imag);
    }

    

    // z_real*z_real + c_real
    *result_real = _mm512_add_ps(r_real, *c_real);

    // z_imag*z_imag + c_imag
    *result_imag = _mm512_add_ps(r_imag, *c_imag);
}

void
fractal_avx512f_z3(__m512 * result_real, __m512 * result_imag, 
                     __m512 * z_real, __m512 * z_imag, 
                     __m512 * c_real, __m512 * c_imag)
{
    fractal_avx512f_zn(result_real, result_imag, z_real, z_imag, c_real, c_imag, 3);
}

void
fractal_avx512f_z4(__m512 * result_real, __m512 * result_imag, 
                     __m512 * z_real, __m512 * z_imag, 
                     __m512 * c_real, __m512 * c_imag)
{
    fractal_avx512f_zn(result_real, result_imag, z_real, z_imag, c_real, c_imag, 4);
}


// Conjugate fractals

void
fractal_avx512f_zconj2(__m512 * result_real, __m512 * result_imag, 
                   __m512 * z_real, __m512 * z_imag, 
                   __m512 * c_real, __m512 * c_imag)
{
    // z * z = (a+bj)*(a+bj) = a*a - b*b + 2(a*b)j
    // conjugate
    *z_imag = _mm512_xor_ps(*z_imag, _mm512_set1_ps(-0.0f));

    _mm512_autocmul_ps(result_real, result_imag, z_real, z_imag);

    // z_real*z_real + c_real
    *result_real = _mm512_add_ps(*result_real, *c_real);

    // z_imag*z_imag + c_imag
    *result_imag = _mm512_add_ps(*result_imag, *c_imag);
}

void
fractal_avx512f_zconjn(__m512 * result_real, __m512 * result_imag, 
                    __m512 * z_real, __m512 * z_imag, 
                    __m512 * c_real, __m512 * c_imag,
                    int n)
{
    // conjugate
    *z_imag = _mm512_xor_ps(*z_imag, _mm512_set1_ps(-0.0f));

    __m512 r_real = *z_real;
    __m512 r_imag = *z_imag;
    for (int i=1; i<n; ++i) {
        _mm512_cmul_ps(&r_real, &r_imag,
                       &r_real, &r_imag,
                       z_real, z_imag);
    }
    // z_real*z_real + c_real
    *result_real = _mm512_add_ps(r_real, *c_real);

    // z_imag*z_imag + c_imag
    *result_imag = _mm512_add_ps(r_imag, *c_imag);
}

void
fractal_avx512f_zconj3(__m512 * result_real, __m512 * result_imag, 
                    __m512 * z_real, __m512 * z_imag, 
                    __m512 * c_real, __m512 * c_imag)
{
    fractal_avx512f_zconjn(result_real, result_imag, z_real, z_imag, c_real, c_imag, 3);
}

void
fractal_avx512f_zconj4(__m512 * result_real, __m512 * result_imag, 
                    __m512 * z_real, __m512 * z_imag, 
                    __m512 * c_real, __m512 * c_imag)
{
    fractal_avx512f_zconjn(result_real, result_imag, z_real, z_imag, c_real, c_imag, 4);
}


// Absolute value fractals

void
fractal_avx512f_zabs2(__m512 * result_real, __m512 * result_imag, 
                    __m512 * z_real, __m512 * z_imag, 
                    __m512 * c_real, __m512 * c_imag)
{
    // z * z = (a+bj)*(a+bj) = a*a - b*b + 2(a*b)j
    // abs
    __m512 sign_mask = _mm512_set1_ps(-0.0f);
    *z_imag = _mm512_andnot_ps(sign_mask, *z_imag);
    *z_real = _mm512_andnot_ps(sign_mask, *z_real);

    _mm512_autocmul_ps(result_real, result_imag, z_real, z_imag);

    // z_real*z_real + c_real
    *result_real = _mm512_add_ps(*result_real, *c_real);

    // z_imag*z_imag + c_imag
    *result_imag = _mm512_add_ps(*result_imag, *c_imag);
}

void
fractal_avx512f_zabsn(__m512 * result_real, __m512 * result_imag, 
                    __m512 * z_real, __m512 * z_imag, 
                    __m512 * c_real, __m512 * c_imag,
                    int n)
{
    // abs
    __m512 sign_mask = _mm512_set1_ps(-0.0f);
    *z_imag = _mm512_andnot_ps(sign_mask, *z_imag);
    *z_real = _mm512_andnot_ps(sign_mask, *z_real);

    __m512 r_real = *z_real;
    __m512 r_imag = *z_imag;

    for (int i=1; i<n; ++i) {
        _mm512_cmul_ps(&r_real, &r_imag,
                       &r_real, &r_imag,
                       z_real, z_imag);
    }
    // z_real*z_real + c_real
    *result_real = _mm512_add_ps(r_real, *c_real);

    // z_imag*z_imag + c_imag
    *result_imag = _mm512_add_ps(r_imag, *c_imag);
}

void
fractal_avx512f_zabs3(__m512 * result_real, __m512 * result_imag, 
                    __m512 * z_real, __m512 * z_imag, 
                    __m512 * c_real, __m512 * c_imag)
{
    fractal_avx512f_zabsn(result_real, result_imag, z_real, z_imag, c_real, c_imag, 3);
}

void
fractal_avx512f_zabs4(__m512 * result_real, __m512 * result_imag, 
                    __m512 * z_real, __m512 * z_imag, 
                    __m512 * c_real, __m512 * c_imag)
{
    fractal_avx512f_zabsn(result_real, result_imag, z_real, z_imag, c_real, c_imag, 4);
}


// Magnet fractals

void
fractal_avx512f_magnet(__m512 * result_real, __m512 * result_imag, 
                    __m512 * z_real, __m512 * z_imag, 
                    __m512 * c_real, __m512 * c_imag)
{
    /* NUMERATOR */
    // z^2
    __m512 z_sq_real, z_sq_imag;
    _mm512_autocmul_ps(&z_sq_real, &z_sq_imag, z_real, z_imag);

    // z^2 + c - 1 = numerator
    __m512 numerator_real = _mm512_add_ps(
        _mm512_add_ps(z_sq_real, *c_real),
        _mm512_set1_ps(-1.0f)
    );
    __m512 numerator_imag = _mm512_add_ps(z_sq_imag, *c_imag);

    /* DENOMINATOR */
    // 2z
    __m512 two = _mm512_set1_ps(2.0f);
    __m512 _2z_real = _mm512_mul_ps(*z_real, two);
    __m512 _2z_imag = _mm512_mul_ps(*z_imag, two);

    // 2z + c - 2
    __m512 denominator_real = _mm512_add_ps(
        _mm512_add_ps(_2z_real, *c_real),
        _mm512_set1_ps(-2.0f)
    );
    __m512 denominator_imag = _mm512_add_ps(_2z_imag, *c_imag);

    /* FRACTION */
    // (a+bj)/(c+dj) = [(a+bj)*(c-dj)]/[(c+dj)*(c-dj)]
    __m512 fraction_real, fraction_imag;
    _mm512_cdiv_ps(&fraction_real, &fraction_imag,
                   &numerator_real, &numerator_imag,
                   &denominator_real, &denominator_imag);

    /* SQUARED FRACTION */
    _mm512_autocmul_ps(result_real, result_imag, &fraction_real, &fraction_imag);
}

// Julia variant z^2 + c/z

void
fractal_avx512f_z2_z(__m512 * result_real, __m512 * result_imag, 
                  __m512 * z_real, __m512 * z_imag, 
                  __m512 * c_real, __m512 * c_imag)
{
    // z * z = (a+bj)*(a+bj) = a*a - b*b + 2(a*b)j
    _mm512_autocmul_ps(result_real, result_imag, z_real, z_imag);

    // c / z
    __m512 fraction_real, fraction_imag;
    _mm512_cdiv_ps(&fraction_real, &fraction_imag,
                   c_real, c_imag,
                   z_real, z_imag);

    // z_real*z_real + c_real/z
    *result_real = _mm512_add_ps(*result_real, fraction_real);

    // z_imag*z_imag + c_imag
    *result_imag = _mm512_add_ps(*result_imag, fraction_imag);
}

static fractal_avx512_t fractalfuncs[FC_FRAC_NUM_ENTRIES] = {
    fractal_avx512f_z2,     // z^2 + c
    fractal_avx512f_z3,     // z^3 + c
    fractal_avx512f_z4,     // z^4 + c
    fractal_avx512f_zconj2, // (conj(z))^2 + c
    fractal_avx512f_zconj3, // (conj(z))^3 + c
    fractal_avx512f_zconj4, // (conj(z))^4 + c
    fractal_avx512f_zabs2,  // (abs(z_real) + abs(c_real)*j)^2 + c
    fractal_avx512f_zabs3,  // (abs(z_real) + abs(c_real)*j)^3 + c
    fractal_avx512f_zabs4,  // (abs(z_real) + abs(c_real)*j)^4 + c
    fractal_avx512f_magnet, // [(z^2 + c - 1)/(2z + c - 2)]^2
    fractal_avx512f_z2_z,
};

fractal_avx512_t fractal_avx512_get(enum FC_Fractal frac)
{
    return fractalfuncs[frac % FC_FRAC_NUM_ENTRIES];
}

#endif // __AVX512DQ__
