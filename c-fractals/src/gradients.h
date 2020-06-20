#ifndef __GRADIENTS_H__
#define __GRADIENTS_H__


#include "fractal_color.h"


/* convert a fractal value to rgb - ultra */
void value_to_rgb_ultra(float * r, float * g, float * b, int value);

/* convert a fractal value to rgb - monochrome */
void value_to_rgb_monochrome(float * r, float * g, float * b, int value);

/* convert a fractal value to rgb - tri */
void value_to_rgb_tri(float * r, float * g, float * b, int value);

/* convert a fractal value to rgb - jet */
void value_to_rgb_jet(float * r, float * g, float * b, int value);

/* get a color function */
void (*colorfunc_get(enum Color color))(float * r, float * g, float * b, int value);


#endif
