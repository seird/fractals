#include "fractals.h"


FRACDTYPE _Complex
fractal_julia(FRACDTYPE _Complex z, FRACDTYPE _Complex c)
{
    return z*z + c;
}


FRACDTYPE _Complex (*fractal_get(enum Fractal frac))(FRACDTYPE _Complex, FRACDTYPE _Complex)
{
    FRACDTYPE _Complex (*fptr)(FRACDTYPE _Complex, FRACDTYPE _Complex) = &fractal_julia;
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
