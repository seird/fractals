#include "fractals.h"


FRACDTYPE _Complex
fractal_z2(FRACDTYPE _Complex z, FRACDTYPE _Complex c)
{
    return z*z + c;
}

FRACDTYPE _Complex
fractal_zn(FRACDTYPE _Complex z, FRACDTYPE _Complex c, int n)
{
    FRACDTYPE _Complex r = z;
    for (int i=1; i<n; ++i) { // (cpowf is ~5x slower for small n..)
        r = r*z;
    }
    return r + c;
}

FRACDTYPE _Complex
fractal_z3(FRACDTYPE _Complex z, FRACDTYPE _Complex c)
{
    return fractal_zn(z, c, 3);
}

FRACDTYPE _Complex
fractal_z4(FRACDTYPE _Complex z, FRACDTYPE _Complex c)
{
    return fractal_zn(z, c, 4);
}

FRACDTYPE _Complex
fractal_zconj2(FRACDTYPE _Complex z, FRACDTYPE _Complex c)
{
    FRACDTYPE _Complex zconj = conjf(z);
    return zconj*zconj + c;
}

FRACDTYPE _Complex
fractal_zconjn(FRACDTYPE _Complex z, FRACDTYPE _Complex c, int n)
{
    FRACDTYPE _Complex r = conjf(z);
    for (int i=1; i<n; ++i) { // (cpowf is ~5x slower for small n..)
        r = r*z;
    }
    return r + c;
}

FRACDTYPE _Complex
fractal_zconj3(FRACDTYPE _Complex z, FRACDTYPE _Complex c)
{
    return fractal_zn(z, c, 3);
}

FRACDTYPE _Complex
fractal_zconj4(FRACDTYPE _Complex z, FRACDTYPE _Complex c)
{
    return fractal_zn(z, c, 4);
}

FRACDTYPE _Complex (*fractal_get(enum Fractal frac))(FRACDTYPE _Complex, FRACDTYPE _Complex)
{
    FRACDTYPE _Complex (*fptr)(FRACDTYPE _Complex, FRACDTYPE _Complex) = &fractal_z2;
    switch (frac) 
    {
        case FRAC_Z2: 
            fptr = &fractal_z2;
            break;
        case FRAC_Z3:
            fptr = &fractal_z3;
            break;
        case FRAC_Z4:
            fptr = &fractal_z4;
            break;
        case FRAC_ZCONJ2: 
            fptr = &fractal_zconj2;
            break;
        case FRAC_ZCONJ3:
            fptr = &fractal_zconj3;
            break;
        case FRAC_ZCONJ4:
            fptr = &fractal_zconj4;
            break;
        default:
            fptr = &fractal_z2;
    }
    return fptr;
}
