/*
    fractal_color.h Library interface
*/

#ifndef __FRACTAL_COLOR_H__
#define __FRACTAL_COLOR_H__


#include <stdint.h>
#include <stdlib.h>


/* Fractal functions */
enum FC_Fractal {
    FC_FRAC_Z2,     // z^2 + c
    FC_FRAC_Z3,     // z^3 + c
    FC_FRAC_Z4,     // z^4 + c
    FC_FRAC_ZCONJ2, // (conj(z))^2 + c
    FC_FRAC_ZCONJ3, // (conj(z))^3 + c
    FC_FRAC_ZCONJ4, // (conj(z))^4 + c
    FC_FRAC_ZABS2,  // (abs(z_real) + abs(c_real)*j)^2 + c
    FC_FRAC_ZABS3,  // (abs(z_real) + abs(c_real)*j)^3 + c
    FC_FRAC_ZABS4,  // (abs(z_real) + abs(c_real)*j)^4 + c
    FC_FRAC_MAGNET, // [(z^2 + c - 1)/(2z + c - 2)]^2
    FC_FRAC_Z2_Z,   // z^2 + c/z
    FC_FRAC_NUM_ENTRIES,
};

/* Fractal modes */
enum FC_Mode {
    FC_MODE_MANDELBROT,
    FC_MODE_JULIA,
    FC_MODE_LYAPUNOV,
    FC_MODE_NUM_ENTRIES,
};

/* Color modes to convert CMATRIX values to */
enum FC_Color {
    FC_COLOR_ULTRA,
    FC_COLOR_MONOCHROME,
    FC_COLOR_TRI,
    FC_COLOR_JET,
    FC_COLOR_LAVENDER,
    FC_COLOR_BINARY,
    FC_COLOR_NUM_ENTRIES
};

/* types */
typedef void * HCMATRIX;

typedef void (* colorfunc_t)(uint8_t * r, uint8_t * g, uint8_t * b, int value);

struct FractalProperties {
    float x_start;
    float x_end;
    float y_start;
    float y_end;
    int width;
    int height;
    enum FC_Fractal frac;
    enum FC_Mode mode;
    float c_real;
    float c_imag;
    float R;
    int max_iterations;
    char * sequence;
    size_t sequence_length;
    float _x_step;
    float _y_step;
};


/* create a color matrix */
HCMATRIX fractal_cmatrix_create(int height, int width);

/* reshape a color matrix */
HCMATRIX fractal_cmatrix_reshape(HCMATRIX hCmatrix, int height_new, int width_new);

/* free a color matrix */
void fractal_cmatrix_free(HCMATRIX hCmatrix);

/* get a matrix value */
float * fractal_cmatrix_value(HCMATRIX hCmatrix, int height, int width);

/* get fractal colors */
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
void fractal_cmatrix_save(HCMATRIX hCmatrix, const char * filename, enum FC_Color color);

/* save an image array as png */
void fractal_image_save(int * image, int width, int height, const char * filename, enum FC_Color color);

/* convert a cmatrix value to rgb */
void fractal_value_to_color(uint8_t * r, uint8_t * g, uint8_t * b, int value, enum FC_Color color);

/* get a color function */
colorfunc_t fractal_colorfunc_get(enum FC_Color color);

/* create an image array */
int * fractal_image_create(int height, int width);

/* free an image array */
void fractal_image_free(int * image);

#endif // __FRACTAL_COLOR_H__
