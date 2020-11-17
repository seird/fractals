#include "gradients.h"


colorfunc_t colorfunc_get(enum Color color)
{
    colorfunc_t fptr = &value_to_rgb_ultra;
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
        case COLOR_JET:
            fptr = &value_to_rgb_jet;
            break;
        case COLOR_NUM_ENTRIES:
        default:
            printf("Unsupported color mode, using COLOR_ULTRA instead.\n");
            fptr = &value_to_rgb_ultra;
    }
    return fptr;
}

void
fractal_value_to_color(float * r, float * g, float * b, int value, enum Color color)
{
    colorfunc_get(color)(r, g, b, value);
}



void
value_to_rgb_jet(float * r, float * g, float * b, int value)
{
    static const float colormap[COLORMAP_SIZE][3] = {
        {  0.00000,     0.00000,   191.25000},
        {  0.00000,     0.00000,   255.00000},
        {  0.00000,    63.75000,   255.00000},
        {  0.00000,   127.50000,   255.00000},
        {  0.00000,   191.25000,   255.00000},
        {  0.00000,   255.00000,   255.00000},
        { 63.75000,   255.00000,   191.25000},
        {127.50000,   255.00000,   127.50000},
        {191.25000,   255.00000,    63.75000},
        {255.00000,   255.00000,     0.00000},
        {255.00000,   191.25000,     0.00000},
        {255.00000,   127.50000,     0.00000},
        {255.00000,    63.75000,     0.00000},
        {255.00000,     0.00000,     0.00000},
        {191.25000,     0.00000,     0.00000},
        {127.50000,     0.00000,     0.00000},
    };

    if (value > 0) {
        *r = colormap[value % 160][0] / 255;
        *g = colormap[value % 160][1] / 255;
        *b = colormap[value % COLORMAP_SIZE*10][2] / 255;
    }
    else {
        *r = *g = *b = 0;
    }
}


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


void
value_to_rgb_tri(float * r, float * g, float * b, int value)
{
    static const float colormap[3][3] = {
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


void
value_to_rgb_ultra(float * r, float * g, float * b, int value)
{
    static const float colormap[COLORMAP_SIZE][3] = {
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
