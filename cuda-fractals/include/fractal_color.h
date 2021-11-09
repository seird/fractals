/*
    fractal_color.h Library interface
*/

#ifndef __FRACTAL_COLOR_H__
#define __FRACTAL_COLOR_H__


#include <stdint.h>
#include <stdlib.h>


/** Fractal functions */
enum FC_Fractal {
    FC_FRAC_Z2,     /**< z^2 + c */
    FC_FRAC_Z3,     /**< z^3 + c */
    FC_FRAC_Z4,     /**< z^4 + c */
    FC_FRAC_ZCONJ2, /**< (conj(z))^2 + c */
    FC_FRAC_ZCONJ3, /**< (conj(z))^3 + c */
    FC_FRAC_ZCONJ4, /**< (conj(z))^4 + c */
    FC_FRAC_ZABS2,  /**< (abs(z_real) + abs(c_real)*j)^2 + c */
    FC_FRAC_ZABS3,  /**< (abs(z_real) + abs(c_real)*j)^3 + c */
    FC_FRAC_ZABS4,  /**< (abs(z_real) + abs(c_real)*j)^4 + c */
    FC_FRAC_MAGNET, /**< [(z^2 + c - 1)/(2z + c - 2)]^2 */
    FC_FRAC_Z2_Z,   /**< z^2 + c/z */
    FC_FRAC_NUM_ENTRIES,
};

/** Fractal modes */
enum FC_Mode {
    FC_MODE_MANDELBROT,
    FC_MODE_JULIA,
    FC_MODE_LYAPUNOV,
    FC_MODE_NUM_ENTRIES,
};

/** Color modes to convert CMATRIX values to */
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
    float x_start;          /**< Lower bound of the real axis in the complex plane */
    float x_end;            /**< Upper bound of the real axis in the complex plane */
    float y_start;          /**< Lower bound of the imaginary axis in the complex plane */
    float y_end;            /**< Upper bound of the imaginary axis in the complex plane */
    int width;              /**< Width of the image (real axis) */
    int height;             /**< Height of the image (imaginary axis) */
    enum FC_Fractal frac;   /**< The fractal function which will be iterated */
    enum FC_Mode mode;      /**< The fractal mode */
    enum FC_Color color;    /**< The color mode to render the fractals (required for cuda) */
    float c_real;           /**< The real part of the c parameter in the fractal function */
    float c_imag;           /**< The imaginary part of the c parameter in the fractal function */
    float R;                /**< Escape radius */
    int max_iterations;     /**< Maximum number of times the fractal function will be iterated */
    char * sequence;        /**< Lyapunov sequence */
    size_t sequence_length; /**< Lyapunov sequence length */
    float _x_step;
    float _y_step;
};


/**
 * @brief Initialize a color matrix, returns a handle @c HCMATRIX
 * 
 * @param height 
 * @param width 
 * @return HCMATRIX 
 */
HCMATRIX fractal_cmatrix_create(int height, int width);

/**
 * @brief Reshape an existing color matrix, returns a handle @c HCMATRIX to the reshaped color matrix. The original color matrix is freed.
 * 
 * @param hCmatrix 
 * @param height_new 
 * @param width_new 
 * @return HCMATRIX 
 */
HCMATRIX fractal_cmatrix_reshape(HCMATRIX hCmatrix, int height_new, int width_new);

/**
 * @brief Free a color matrix
 * 
 * @param hCmatrix 
 */
void fractal_cmatrix_free(HCMATRIX hCmatrix);

/**
 * @brief Retrieve a pointer to a value in the color matrix
 * 
 * @param hCmatrix 
 * @param height 
 * @param width 
 * @return float* 
 */
float * fractal_cmatrix_value(HCMATRIX hCmatrix, int height, int width);

/**
 * @brief Compute the fractal colors
 * 
 * @param hCmatrix 
 * @param fp 
 */
void fractal_get_colors(HCMATRIX hCmatrix, struct FractalProperties * fp);

/**
 * @brief Compute the fractal colors with threads
 * 
 * @param hCmatrix 
 * @param fp 
 * @param num_threads 
 */
void fractal_get_colors_th(HCMATRIX hCmatrix, struct FractalProperties * fp, int num_threads);

/**
 * @brief Compute the fractal colors with AVX2
 * 
 * @param hCmatrix 
 * @param fp 
 */
void fractal_avxf_get_colors(HCMATRIX hCmatrix, struct FractalProperties * fp);

/**
 * @brief Compute the fractal colors with AVX2 and threads
 * 
 * @param hCmatrix 
 * @param fp 
 * @param num_threads 
 */
void fractal_avxf_get_colors_th(HCMATRIX hCmatrix, struct FractalProperties * fp, int num_threads);

/**
 * @brief Retrieve the maximum color value in the color matrix
 * 
 * @param hCmatrix 
 * @return float 
 */
float fractal_cmatrix_max(HCMATRIX hCmatrix);

/**
 * @brief Save a color matrix as png
 * 
 * @param hCmatrix 
 * @param filename 
 * @param color 
 */
void fractal_cmatrix_save(HCMATRIX hCmatrix, const char * filename, enum FC_Color color);

/**
 * @brief Convert a cmatrix value to rgb
 * 
 * @param r 
 * @param g 
 * @param b 
 * @param value 
 * @param color 
 */
void fractal_value_to_color(uint8_t * r, uint8_t * g, uint8_t * b, int value, enum FC_Color color);

/**
 * @brief Get a color function
 * 
 * @param color 
 * @return colorfunc_t 
 */
colorfunc_t fractal_colorfunc_get(enum FC_Color color);

#endif // __FRACTAL_COLOR_H__
