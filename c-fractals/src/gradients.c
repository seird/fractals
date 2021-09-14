#include "gradients.h"


static colorfunc_t colorfuncs[FC_COLOR_NUM_ENTRIES] = {
    fractal_value_to_rgb_ultra,
    fractal_value_to_rgb_monochrome,
    fractal_value_to_rgb_tri,
    fractal_value_to_rgb_jet,
    fractal_value_to_rgb_lavender,
    fractal_value_to_rgb_binary,
};

colorfunc_t fractal_colorfunc_get(enum FC_Color color)
{
    return colorfuncs[color % FC_COLOR_NUM_ENTRIES];
}

void
fractal_value_to_color(float * r, float * g, float * b, int value, enum FC_Color color)
{
    fractal_colorfunc_get(color)(r, g, b, value);
}



void
fractal_value_to_rgb_jet(float * r, float * g, float * b, int value)
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
        *r = colormap[value % 160][0];
        *g = colormap[value % 160][1];
        *b = colormap[value % COLORMAP_SIZE*10][2];
    }
    else {
        *r = *g = *b = 0;
    }
}


void
fractal_value_to_rgb_monochrome(float * r, float * g, float * b, int value)
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
        *r = colormap[value % COLORMAP_SIZE][0];
        *g = colormap[value % COLORMAP_SIZE][1];
        *b = colormap[value % COLORMAP_SIZE][2];
    }
    else {
        *r = *g = *b = 0;
    }
}


void
fractal_value_to_rgb_tri(float * r, float * g, float * b, int value)
{
    static const float colormap[3][3] = {
        {255, 150, 150},
        {150, 255, 150},
        {150, 150, 255},
    };

    if (value > 0) {
        *r = colormap[value % 3][0];
        *g = colormap[value % 3][1];
        *b = colormap[value % 3][2];
    }
    else {
        *r = *g = *b = 0;
    }
}


void
fractal_value_to_rgb_ultra(float * r, float * g, float * b, int value)
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
        *r = colormap[value % COLORMAP_SIZE][0];
        *g = colormap[value % COLORMAP_SIZE][1];
        *b = colormap[value % COLORMAP_SIZE][2];
    }
    else {
        *r = *g = *b = 0;
    }
}


void fractal_value_to_rgb_lavender(float * r, float * g, float * b, int value) {
    static const float colormap[][3] = {
       { 69, 147, 254},
       {101, 154, 214},
       {108, 122, 211},
       {114,  89, 203},
       {119,  80, 183},
       {123, 134, 163},
       {128,  51, 186},
       {132,  88, 151},
       {136, 116, 153},
       {140, 126, 130},
       {144, 145, 111},
       {149,  56, 151},
       {153, 105, 124},
       {158,  23, 131},
       {162, 112, 117},
       {167,  60,  93},
       {172,  42,  83},
       {177,  44, 111},
       {182,  49,  86},
       {187, 163,   7},
       {193,  33,  57},
       {198, 188,  29},
       {204,  75,  91},
       {210,  48,  73},
       {216,  32,  85},
       {221, 118, 128},
       {226, 207,   0},
       {231,  68,  85},
       {235, 169, 169},
       {239, 147, 159},
       {243,  67,  93},
       {246, 179, 193},
       {250,  83, 103},
       {253, 224, 226}
    };

    if (value > 0) {
        *r = colormap[value % sizeof(colormap)/3][0];
        *g = colormap[value % sizeof(colormap)/3][1];
        *b = colormap[value % sizeof(colormap)/3][2];
    } else {
        *r = *g = *b = 0;
    }
}


void
fractal_value_to_rgb_binary(float * r, float * g, float * b, int value)
{
    *r = *g = *b = 255*(value > 0);
}
