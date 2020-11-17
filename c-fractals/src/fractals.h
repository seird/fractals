#ifndef __FRACTALS_H__
#define __FRACTALS_H__


#include <math.h>
#include <complex.h>

#include "fractal_color.h"


typedef float _Complex (*fractal_t)(float _Complex, float _Complex);


float _Complex fractal_z2(float _Complex z, float _Complex c);

float _Complex fractal_zn(float _Complex z, float _Complex c, int n);

float _Complex fractal_z3(float _Complex z, float _Complex c);

float _Complex fractal_z4(float _Complex z, float _Complex c);


fractal_t fractal_get(enum Fractal frac);

#endif
