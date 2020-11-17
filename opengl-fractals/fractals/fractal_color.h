/*
    fractal_color.h Library interface
*/

#ifndef __FRACTAL_COLOR_H__
#define __FRACTAL_COLOR_H__



/* Fractal functions */
enum Fractal {
    FRAC_Z2,     // z^2 + c
    FRAC_Z3,     // z^3 + c
    FRAC_Z4,     // z^4 + c
    FRAC_ZCONJ2, // (conj(z))^2 + c
    FRAC_ZCONJ3, // (conj(z))^3 + c
    FRAC_ZCONJ4, // (conj(z))^4 + c
    FRAC_ZABS2,  // (abs(z_real) + abs(c_real)*j)^2 + c
    FRAC_ZABS3,  // (abs(z_real) + abs(c_real)*j)^3 + c
    FRAC_ZABS4,  // (abs(z_real) + abs(c_real)*j)^4 + c
    FRAC_MAGNET, // [(z^2 + c - 1)/(2z + c - 2)]^2
    FRAC_Z2_Z,   // z^2 + c/z
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
    COLOR_JET,
    COLOR_NUM_ENTRIES
};

/* types */
typedef void * HCMATRIX;

struct FractalProperties {
    float x_start;
    float x_step;
    float y_start;
    float y_step;
    enum Fractal frac;
    enum Mode mode;
    float c_real;
    float c_imag;
    float R;
    int max_iterations;
};


/* create a color matrix */
HCMATRIX fractal_cmatrix_create(int ROWS, int COLS);

/* reshape a color matrix */
HCMATRIX fractal_cmatrix_reshape(HCMATRIX hCmatrix, int ROWS_new, int COLS_new);

/* free a color matrix */
void fractal_cmatrix_free(HCMATRIX hCmatrix);

/* get a matrix value */
float * fractal_cmatrix_value(HCMATRIX hCmatrix, int row, int col);

/* get fractal colors without complex data type */
void fractal_get_colors(HCMATRIX hCmatrix, struct FractalProperties * fp);

/* get fractal colors with threading */
void fractal_get_colors_th(HCMATRIX hCmatrix, struct FractalProperties * fp, int num_threads);

/* get fractal colors with AVX2 */
void fractal_avxf_get_colors(HCMATRIX hCmatrix, struct FractalProperties * fp);

/* get fractal colors with AVX2 and threading */
void fractal_avxf_get_colors_th(HCMATRIX hCmatrix, struct FractalProperties * fp, int num_threads);

/* get the maximum color value */
float fractal_cmatrix_max(HCMATRIX hCmatrix);

/* save a color matrix as png */
void fractal_cmatrix_save(HCMATRIX hCmatrix, const char * filename, enum Color color);

/* convert a cmatrix value to rgb */
void fractal_value_to_color(float * r, float * g, float * b, int value, enum Color color);


#endif
