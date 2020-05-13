/*
    fracal_color.h Library interface
*/

#ifndef __FRACTAL_COLOR_H__
#define __FRACTAL_COLOR_H__


/* Fractal functions */
enum Fractal {
    FRAC_JULIA,
};

/* types */
typedef void * HCMATRIX;


#ifdef COMPLEX
#include <complex.h>
/* get fractal colors */
void fractal_get_colors_cmpx(HCMATRIX hCmatrix,
                        float x_start, float x_step, float y_start, float y_step,
                        enum Fractal frac, float _Complex c,
                        float R, int max_iterations);
#endif

/* create a color matrix */
HCMATRIX fractal_cmatrix_create(int ROWS, int COLS);

/* reshape a color matrix */
HCMATRIX fractal_cmatrix_reshape(HCMATRIX hCmatrix, int ROWS_new, int COLS_new);

/* free a color matrix */
void fractal_cmatrix_free(HCMATRIX hCmatrix);

/* get a matrix value */
float * fractal_cmatrix_value(HCMATRIX hCmatrix, int row, int col);

/* get fractal colors without complex data type */
void fractal_get_colors(HCMATRIX hCmatrix,
                           float x_start, float x_step, float y_start, float y_step,
                           enum Fractal frac, float c_real, float c_imag,
                           float R, int max_iterations);

/* get fractal colors with threading */
void fractal_get_colors_th(HCMATRIX hCmatrix,
                           float x_start, float x_step, float y_start, float y_step,
                           enum Fractal frac, float c_real, float c_imag,
                           float R, int max_iterations, int num_threads);

/* get the maximum color value */
float fractal_get_max_color(HCMATRIX hCmatrix);

#endif
