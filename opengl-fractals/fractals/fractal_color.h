/*
    fracal_color.h Library interface
*/

#ifndef __FRACTAL_COLOR_H__
#define __FRACTAL_COLOR_H__


#define FRACDTYPE double


/* Fractal functions */
enum Fractal {
    FRAC_JULIA,
};

/* types */
typedef void * HCMATRIX;

struct FractalProperties {
    FRACDTYPE x_start;
    FRACDTYPE x_step;
    FRACDTYPE y_start;
    FRACDTYPE y_step;
    enum Fractal frac;
    FRACDTYPE c_real;
    FRACDTYPE c_imag;
    FRACDTYPE R;
    int max_iterations;
};


/* create a color matrix */
HCMATRIX fractal_cmatrix_create(int ROWS, int COLS);

/* reshape a color matrix */
HCMATRIX fractal_cmatrix_reshape(HCMATRIX hCmatrix, int ROWS_new, int COLS_new);

/* free a color matrix */
void fractal_cmatrix_free(HCMATRIX hCmatrix);

/* get a matrix value */
FRACDTYPE * fractal_cmatrix_value(HCMATRIX hCmatrix, int row, int col);

/* get fractal colors without complex data type */
void fractal_get_colors(HCMATRIX hCmatrix, struct FractalProperties * fp);

/* get fractal colors with threading */
void fractal_get_colors_th(HCMATRIX hCmatrix, struct FractalProperties * fp, int num_threads);

/* get the maximum color value */
FRACDTYPE fractal_get_max_color(HCMATRIX hCmatrix);

#endif
