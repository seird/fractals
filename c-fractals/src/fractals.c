#include "fractals.h"


// Regular fractals: z^2

float _Complex
fractal_z2(float _Complex z, float _Complex c)
{
    return z*z + c;
}

float _Complex
fractal_zn(float _Complex z, float _Complex c, int n)
{
    float _Complex r = z;
    for (int i=1; i<n; ++i) { // (cpowf is ~5x slower for small n..)
        r = r*z;
    }
    return r + c;
}

float _Complex
fractal_z3(float _Complex z, float _Complex c)
{
    return fractal_zn(z, c, 3);
}

float _Complex
fractal_z4(float _Complex z, float _Complex c)
{
    return fractal_zn(z, c, 4);
}

fractal_t fractal_get(enum Fractal frac)
{
    fractal_t fptr = &fractal_z2;
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
