#include "fractals.h"

// Regular fractals: z^2

float _Complex fractal_z2(float _Complex z, float _Complex c)
{
    return z * z + c;
}

float _Complex fractal_zn(float _Complex z, float _Complex c, int n)
{
    float _Complex r = z;
    for (int i = 1; i < n; ++i) { // (cpowf is ~5x slower for small n..)
        r = r * z;
    }
    return r + c;
}

float _Complex fractal_z3(float _Complex z, float _Complex c)
{
    return fractal_zn(z, c, 3);
}

float _Complex fractal_z4(float _Complex z, float _Complex c)
{
    return fractal_zn(z, c, 4);
}

// Conjugate fractals

float _Complex fractal_zconj2(float _Complex z, float _Complex c)
{
    return fractal_z2(conjf(z), c);
}

float _Complex fractal_zconj3(float _Complex z, float _Complex c)
{
    return fractal_z3(conjf(z), c);
}

float _Complex fractal_zconj4(float _Complex z, float _Complex c)
{
    return fractal_z4(conjf(z), c);
}

float _Complex fractal_zconjn(float _Complex z, float _Complex c, int n)
{
    return fractal_zn(conjf(z), c, n);
}

// Absolute value fractals

float _Complex fractal_zabs2(float _Complex z, float _Complex c)
{
    return fractal_z2(fabsf(crealf(z)) + fabsf(crealf(c)) * I, c);
}

float _Complex fractal_zabs3(float _Complex z, float _Complex c)
{
    return fractal_z3(fabsf(crealf(z)) + fabsf(crealf(c)) * I, c);
}

float _Complex fractal_zabs4(float _Complex z, float _Complex c)
{
    return fractal_z4(fabsf(crealf(z)) + fabsf(crealf(c)) * I, c);
}

float _Complex fractal_zabsn(float _Complex z, float _Complex c, int n)
{
    return fractal_zn(fabsf(crealf(z)) + fabsf(crealf(c)) * I, c, n);
}

fractal_t
fractal_get(enum FC_Fractal frac)
{
    fractal_t fptr = &fractal_z2;
    switch (frac) {
        case FC_FRAC_Z2:
            fptr = &fractal_z2;
            break;
        case FC_FRAC_Z3:
            fptr = &fractal_z3;
            break;
        case FC_FRAC_Z4:
            fptr = &fractal_z4;
            break;
        case FC_FRAC_ZCONJ2:
            fptr = &fractal_zconj2;
            break;
        case FC_FRAC_ZCONJ3:
            fptr = &fractal_zconj3;
            break;
        case FC_FRAC_ZCONJ4:
            fptr = &fractal_zconj4;
            break;
        case FC_FRAC_ZABS2:
            fptr = &fractal_zabs2;
            break;
        case FC_FRAC_ZABS3:
            fptr = &fractal_zabs3;
            break;
        case FC_FRAC_ZABS4:
            fptr = &fractal_zabs4;
            break;
        default:
            fptr = &fractal_z2;
    }
    return fptr;
}
