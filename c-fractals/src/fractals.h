#ifndef __FRACTALS_H__
#define __FRACTALS_H__

#include <complex.h>
#include <math.h>

#include "../include/fractal_color.h"

typedef float _Complex (*fractal_t)(float _Complex, float _Complex);

// Regular fractals

float _Complex fractal_z2(float _Complex z, float _Complex c);
float _Complex fractal_zn(float _Complex z, float _Complex c, int n);
float _Complex fractal_z3(float _Complex z, float _Complex c);
float _Complex fractal_z4(float _Complex z, float _Complex c);

// Conjugate fractals

float _Complex fractal_zconj2(float _Complex z, float _Complex c);
float _Complex fractal_zconj3(float _Complex z, float _Complex c);
float _Complex fractal_zconj4(float _Complex z, float _Complex c);
float _Complex fractal_zconjn(float _Complex z, float _Complex c, int n);

// Absolute value fractals

float _Complex fractal_zabs2(float _Complex z, float _Complex c);
float _Complex fractal_zabs3(float _Complex z, float _Complex c);
float _Complex fractal_zabs4(float _Complex z, float _Complex c);
float _Complex fractal_zabsn(float _Complex z, float _Complex c, int n);

fractal_t
fractal_get(enum FC_Fractal frac);

#endif
