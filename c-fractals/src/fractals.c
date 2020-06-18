#include "fractals.h"


// Regular fractals: z^2

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
        default:
            fptr = &fractal_z2;
    }
    return fptr;
}
