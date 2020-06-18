#ifndef __FRACTALS_H__
#define __FRACTALS_H__


#include <math.h>
#include <complex.h>

#include "fractal_color.h"



FRACDTYPE _Complex fractal_z2(FRACDTYPE _Complex z, FRACDTYPE _Complex c);

FRACDTYPE _Complex fractal_zn(FRACDTYPE _Complex z, FRACDTYPE _Complex c, int n);

FRACDTYPE _Complex fractal_z3(FRACDTYPE _Complex z, FRACDTYPE _Complex c);

FRACDTYPE _Complex fractal_z4(FRACDTYPE _Complex z, FRACDTYPE _Complex c);

FRACDTYPE _Complex fractal_zconj2(FRACDTYPE _Complex z, FRACDTYPE _Complex c);

FRACDTYPE _Complex fractal_zconjn(FRACDTYPE _Complex z, FRACDTYPE _Complex c, int n);

FRACDTYPE _Complex fractal_zconj3(FRACDTYPE _Complex z, FRACDTYPE _Complex c);

FRACDTYPE _Complex fractal_zconj4(FRACDTYPE _Complex z, FRACDTYPE _Complex c);


FRACDTYPE _Complex (*fractal_get(enum Fractal frac))(FRACDTYPE _Complex, FRACDTYPE _Complex);

#endif
