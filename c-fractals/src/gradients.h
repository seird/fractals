#ifndef __GRADIENTS_H__
#define __GRADIENTS_H__


#include <stdio.h>

#include "../include/fractal_color.h"


#define COLORMAP_SIZE 16


/* convert a fractal value to rgb - ultra */
void fractal_value_to_rgb_ultra(uint8_t * r, uint8_t * g, uint8_t * b, int value);

/* convert a fractal value to rgb - monochrome */
void fractal_value_to_rgb_monochrome(uint8_t * r, uint8_t * g, uint8_t * b, int value);

/* convert a fractal value to rgb - tri */
void fractal_value_to_rgb_tri(uint8_t * r, uint8_t * g, uint8_t * b, int value);

/* convert a fractal value to rgb - jet */
void fractal_value_to_rgb_jet(uint8_t * r, uint8_t * g, uint8_t * b, int value);

/* convert a fractal value to rgb - lavender */
void fractal_value_to_rgb_lavender(uint8_t * r, uint8_t * g, uint8_t * b, int value);

/* convert a fractal value to rgb - binary */
void fractal_value_to_rgb_binary(uint8_t * r, uint8_t * g, uint8_t * b, int value);


#endif
