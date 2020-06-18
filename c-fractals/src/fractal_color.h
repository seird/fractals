/*
    fractal_color.h Library interface
*/

#ifndef __FRACTAL_COLOR_H__
#define __FRACTAL_COLOR_H__


#ifndef FRACDTYPE
#define FRACDTYPE float
#endif


/* Fractal functions */
enum Fractal {
    FRAC_Z2,     // z^2 + c
    FRAC_Z3,     // z^3 + c
    FRAC_Z4,     // z^4 + c
    FRAC_ZCONJ2, // (conj(z))^2 + c
    FRAC_ZCONJ3, // (conj(z))^3 + c
    FRAC_ZCONJ4, // (conj(z))^4 + c
};

/* Fractal modes */
enum Mode {
    MODE_MANDELBROT,
    MODE_JULIA,
    MODE_BUDDHABROT,
};

/* Color modes to convert CMATRIX values to */
enum Color {
    COLOR_ULTRA,
    COLOR_MONOCHROME,
    COLOR_TRI,
};

/* types */
typedef void * HCMATRIX;

struct FractalProperties {
    FRACDTYPE x_start;
    FRACDTYPE x_step;
    FRACDTYPE y_start;
    FRACDTYPE y_step;
    enum Fractal frac;
    enum Mode mode;
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

/* get fractal colors with AVX2 */
void fractal_avxf_get_colors(HCMATRIX hCmatrix, struct FractalProperties * fp);

/* get fractal colors with AVX2 and threading */
void fractal_avxf_get_colors_th(HCMATRIX hCmatrix, struct FractalProperties * fp, int num_threads);

/* get the maximum color value */
FRACDTYPE fractal_cmatrix_max(HCMATRIX hCmatrix);

/* save a color matrix as png */
void fractal_cmatrix_save(HCMATRIX hCmatrix, const char * filename, enum Color color);

/* convert a cmatrix value to rgb */
void fractal_value_to_color(float * r, float * g, float * b, int value, enum Color color);


#endif
