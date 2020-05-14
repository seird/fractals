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


typedef FRACDTYPE ** CMATRIX;
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


bool escape_magnitude_check(FRACDTYPE _Complex z, FRACDTYPE R);
void fractal_get_single_color(FRACDTYPE * color, FRACDTYPE x, FRACDTYPE y, FRACDTYPE _Complex (*fractal)(FRACDTYPE complex, FRACDTYPE _Complex), FRACDTYPE _Complex c, FRACDTYPE R, int max_iterations);
void * get_colors_thread_worker(void * arg);

#if (!defined(TEST) && !defined(SHARED))
int main(int argc, char * argv[]);
#endif


#endif
