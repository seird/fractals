/*
    fractal_color.h Library interface
*/

#ifndef __FRACTAL_COLOR_H__
#define __FRACTAL_COLOR_H__


#define FRACDTYPE float
#define COLORMAP_SIZE 16


/* Fractal functions */
enum Fractal {
    FRAC_JULIA,   // z^2 + c
    FRAC_JULIA_3, // z^3 + c
    FRAC_JULIA_4, // z^4 + c
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

/* get fractal colors with AVX2 */
void fractal_avxf_get_colors(HCMATRIX hCmatrix, struct FractalProperties * fp);

/* get fractal colors with AVX2 and threading */
void fractal_avxf_get_colors_th(HCMATRIX hCmatrix, struct FractalProperties * fp, int num_threads);

/* get the maximum color value */
FRACDTYPE fractal_cmatrix_max(HCMATRIX hCmatrix);

/* save a color matrix as jpg */
void fractal_cmatrix_save(HCMATRIX hCmatrix, const char * filename);

/* convert a fractal value to rgb - ultra */
void value_to_rgb_ultra(float * r, float * g, float * b, int value);

/* convert a fractal value to rgb - monochrome */
void value_to_rgb_monochrome(float * r, float * g, float * b, int value);

/* convert a fractal value to rgb - tri */
void value_to_rgb_tri(float * r, float * g, float * b, int value);

#endif
