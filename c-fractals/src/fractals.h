#ifndef __FRACTALS_H__
#define __FRACTALS_H__


#include <math.h>
#include <complex.h>

#include "fractal_color.h"



FRACDTYPE _Complex fractal_julia(FRACDTYPE _Complex z, FRACDTYPE _Complex c);

FRACDTYPE _Complex fractal_julia_n(FRACDTYPE _Complex z, FRACDTYPE _Complex c, int n);

FRACDTYPE _Complex fractal_julia_3(FRACDTYPE _Complex z, FRACDTYPE _Complex c);

FRACDTYPE _Complex fractal_julia_4(FRACDTYPE _Complex z, FRACDTYPE _Complex c);


FRACDTYPE _Complex (*fractal_get(enum Fractal frac))(FRACDTYPE _Complex, FRACDTYPE _Complex);

#endif
