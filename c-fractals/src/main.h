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
#include <immintrin.h>

#include "fractals.h"
#include "fractal_color.h"


#define BLACK 0.0


typedef float ** CMATRIX;
typedef struct S_CMATRIX {
    CMATRIX cmatrix;
    int ROWS;
    int COLS;
} * HS_CMATRIX;

struct ThreadArg {
    HS_CMATRIX hc;
    int row_start;
    int row_end;
    struct FractalProperties * fp;
};


bool escape_magnitude_check(float _Complex z, float R);
void fractal_get_single_color(float * color, float x, float y, float _Complex (*fractal)(float complex, float _Complex), float _Complex c, float R, int max_iterations);
void * get_colors_thread_worker(void * arg);

#if (!defined(TEST) && !defined(SHARED))
int main(int argc, char * argv[]);
#endif


#endif
