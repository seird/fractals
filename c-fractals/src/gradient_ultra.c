#include "main.h"


void
value_to_rgb_ultra(float * r, float * g, float * b, int value)
{
    float colormap[COLORMAP_SIZE][3] = {
        {66, 30, 15},
        {25, 7, 26},
        {9, 1, 47},
        {4, 4, 73},
        {0, 7, 100},
        {12, 44, 138},
        {24, 82, 177},
        {57, 125, 209},
        {134, 181, 229},
        {211, 236, 248},
        {241, 233, 191},
        {248, 201, 95},
        {255, 170, 0},
        {204, 128, 0},
        {153, 87, 0},
        {106, 52, 3},
    };

    if (value > 0) {
        *r = colormap[value % COLORMAP_SIZE][0]/255;
        *g = colormap[value % COLORMAP_SIZE][1]/255;
        *b = colormap[value % COLORMAP_SIZE][2]/255;
    }
    else {
        *r = *g = *b = 0;
    }
}
