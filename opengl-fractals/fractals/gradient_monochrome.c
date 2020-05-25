#include "gradients.h"


void
value_to_rgb_monochrome(float * r, float * g, float * b, int value, float max_color)
{
    *r = *g = *b = value / max_color;
}
