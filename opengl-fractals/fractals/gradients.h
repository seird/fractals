#ifndef __GRADIENTS_H__
#define __GRADIENTS_H__


#define COLORMAP_SIZE 16



void value_to_rgb_ultra(float * r, float * g, float * b, int value);
void value_to_rgb_monochrome(float * r, float * g, float * b, int value, float max_color);


#endif
