#include "main.h"


void
value_to_rgb_monochrome(float * r, float * g, float * b, int value)
{
    static const float colormap[COLORMAP_SIZE][3] = {
        {50, 50, 50},
        {60, 60, 60},
        {70, 70, 70},
        {80, 80, 80},
        {90, 90, 90},
        {100, 100, 100},
        {110, 110, 110},
        {120, 120, 120},
        {130, 130, 130},
        {140, 140, 140},
        {150, 150, 150},
        {160, 160, 160},
        {170, 170, 170},
        {180, 180, 180},
        {190, 190, 190},
    };

    if (value > 0) {
        *r = colormap[value % COLORMAP_SIZE][0] / 255;
        *g = colormap[value % COLORMAP_SIZE][1] / 255;
        *b = colormap[value % COLORMAP_SIZE][2] / 255;
    }
    else {
        *r = *g = *b = 0;
    }
}
