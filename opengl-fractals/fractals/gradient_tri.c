#include "gradients.h"


void
value_to_rgb_tri(float * r, float * g, float * b, int value)
{
    float colormap[3][3] = {
        {255, 150, 150},
        {150, 255, 150},
        {150, 150, 255},
    };

    if (value > 0) {
        *r = colormap[value % 3][0] / 255;
        *g = colormap[value % 3][1] / 255;
        *b = colormap[value % 3][2] / 255;
    }
    else {
        *r = *g = *b = 0;
    }
}