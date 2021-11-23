#ifndef __MAIN_H__
#define __MAIN_H__


//#include <windows.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>
#include <complex.h>
#include <stdbool.h>
#include <pthread.h>
#include <string.h>

#include "gradients.h"
#include "fractals.h"
#include "../include/fractal_color.h"
#include "compute_avx.h"
#include "flames/flames.h"


#define BLACK 0.0


typedef float ** CMATRIX;
typedef struct S_CMATRIX {
    CMATRIX cmatrix;
    int height;
    int width;
} * HS_CMATRIX;

struct ThreadArg {
    int thread_id;
    int num_threads;
    HS_CMATRIX hc;
    struct FractalProperties * fp;
};


bool fractal_escape_magnitude_check(float _Complex z, float R);
void fractal_get_single_color(float * color, float x, float y, fractal_t fractal, float _Complex c, float R, int max_iterations);

#if (!defined(TEST) && !defined(SHARED) && !defined(BENCHMARK))
int main(void);
#endif


#endif
