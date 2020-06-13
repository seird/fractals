#include "gradients.h"


void (*colorfunc_get(enum Color color))(float * r, float * g, float * b, int value)
{
    void (*fptr)(float * r, float * g, float * b, int value) = &value_to_rgb_ultra;
    switch (color) 
    {
        case COLOR_ULTRA: 
            fptr = &value_to_rgb_ultra;
            break;
        case COLOR_MONOCHROME:
            fptr = &value_to_rgb_monochrome;
            break;
        case COLOR_TRI:
            fptr = &value_to_rgb_tri;
            break;
        default:
            fptr = &value_to_rgb_ultra;
    }
    return fptr;
}

void
fractal_value_to_color(float * r, float * g, float * b, int value, enum Color color)
{
    colorfunc_get(color)(r, g, b, value);
}
