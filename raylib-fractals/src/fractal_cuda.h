#pragma once


#include <stdbool.h>
#include "fractal_color.h"

#ifdef CUDA
extern bool fractal_cuda_init(int width, int height);
extern void fractal_cuda_clean();
extern void fractal_cuda_get_colors(int * image, struct FractalProperties * fp);
#endif
