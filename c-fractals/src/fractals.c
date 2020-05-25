#include "fractals.h"


FRACDTYPE _Complex
fractal_julia(FRACDTYPE _Complex z, FRACDTYPE _Complex c)
{
    return z*z + c;
}

FRACDTYPE _Complex
fractal_julia_n(FRACDTYPE _Complex z, FRACDTYPE _Complex c, int n)
{
    FRACDTYPE _Complex r = z;
    for (int i=1; i<n; ++i) { // (cpowf is ~5x slower for small n..)
        r = r*z;
    }
    return r + c;
}

FRACDTYPE _Complex
fractal_julia_3(FRACDTYPE _Complex z, FRACDTYPE _Complex c)
{
    return fractal_julia_n(z, c, 3);
}

FRACDTYPE _Complex
fractal_julia_4(FRACDTYPE _Complex z, FRACDTYPE _Complex c)
{
    return fractal_julia_n(z, c, 4);
}


FRACDTYPE _Complex (*fractal_get(enum Fractal frac))(FRACDTYPE _Complex, FRACDTYPE _Complex)
{
    FRACDTYPE _Complex (*fptr)(FRACDTYPE _Complex, FRACDTYPE _Complex) = &fractal_julia;
    switch (frac) 
    {
        case FRAC_JULIA: 
            fptr = &fractal_julia;
            break;
        case FRAC_JULIA_3:
            fptr = &fractal_julia_3;
            break;
        case FRAC_JULIA_4:
            fptr = &fractal_julia_4;
            break;
        default:
            fptr = &fractal_julia;
    }
    return fptr;
}
