#include "fractals.h"


double _Complex
fractal_julia(double _Complex z, double _Complex c)
{
    return z*z + c;
}


double _Complex (*fractal_get(enum Fractal frac))(double _Complex, double _Complex)
{
    double _Complex (*fptr)(double _Complex, double _Complex) = &fractal_julia;
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
