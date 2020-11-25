#ifndef __GRADIENTS_H__
#define __GRADIENTS_H__


#include <stdio.h>

#include "fractal_color.h"


#define COLORMAP_SIZE 16


typedef void (* colorfunc_t)(float * r, float * g, float * b, int value);


/* convert a fractal value to rgb - ultra */
void value_to_rgb_ultra(float * r, float * g, float * b, int value);

/* convert a fractal value to rgb - monochrome */
void value_to_rgb_monochrome(float * r, float * g, float * b, int value);

/* convert a fractal value to rgb - tri */
void value_to_rgb_tri(float * r, float * g, float * b, int value);

/* convert a fractal value to rgb - jet */
void value_to_rgb_jet(float * r, float * g, float * b, int value);

/* convert a fractal value to rgb - lavender */
void value_to_rgb_lavender(float * r, float * g, float * b, int value);

/* get a color function */
colorfunc_t colorfunc_get(enum FC_Color color);


#endif
