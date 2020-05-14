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


typedef double ** CMATRIX;
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


bool escape_magnitude_check(double _Complex z, double R);
void fractal_get_single_color(double * color, double x, double y, double _Complex (*fractal)(double complex, double _Complex), double _Complex c, double R, int max_iterations);
void * get_colors_thread_worker(void * arg);

#if (!defined(TEST) && !defined(SHARED))
int main(int argc, char * argv[]);
#endif


#endif
