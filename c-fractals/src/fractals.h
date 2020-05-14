#ifndef __FRACTALS_H__
#define __FRACTALS_H__


#include <math.h>
#include <complex.h>

#include "fractal_color.h"



double _Complex fractal_julia(double _Complex z, double _Complex c);

double _Complex (*fractal_get(enum Fractal frac))(double _Complex, double _Complex);

#endif
