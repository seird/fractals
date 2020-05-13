#ifndef __FRACTALS_H__
#define __FRACTALS_H__


#include <math.h>
#include <complex.h>

#include "fractal_color.h"



float _Complex fractal_julia(float _Complex z, float _Complex c);

float _Complex (*fractal_get(enum Fractal frac))(float _Complex, float _Complex);

#endif
