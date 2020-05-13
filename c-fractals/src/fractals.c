#include "fractals.h"


float _Complex
fractal_julia(float _Complex z, float _Complex c)
{
    return z*z + c;
}


float _Complex (*fractal_get(enum Fractal frac))(float _Complex, float _Complex)
{
    float _Complex (*fptr)(float _Complex, float _Complex) = &fractal_julia;
    switch (frac) 
    {
        case FRAC_JULIA: 
            fptr = &fractal_julia;
            break;
        default:
            fptr = &fractal_julia;
    }
    return fptr;
}
