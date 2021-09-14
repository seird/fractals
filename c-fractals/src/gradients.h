#ifndef __GRADIENTS_H__
#define __GRADIENTS_H__


#include <stdio.h>

#include "../include/fractal_color.h"


#define COLORMAP_SIZE 16


/* convert a fractal value to rgb - ultra */
void fractal_value_to_rgb_ultra(float * r, float * g, float * b, int value);

/* convert a fractal value to rgb - monochrome */
void fractal_value_to_rgb_monochrome(float * r, float * g, float * b, int value);

/* convert a fractal value to rgb - tri */
void fractal_value_to_rgb_tri(float * r, float * g, float * b, int value);

/* convert a fractal value to rgb - jet */
void fractal_value_to_rgb_jet(float * r, float * g, float * b, int value);

/* convert a fractal value to rgb - lavender */
void fractal_value_to_rgb_lavender(float * r, float * g, float * b, int value);

/* convert a fractal value to rgb - binary */
void fractal_value_to_rgb_binary(float * r, float * g, float * b, int value);


#endif
