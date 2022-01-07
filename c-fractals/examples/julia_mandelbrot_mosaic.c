/* Generate a Mandelbrot mosaic consisting of small Julia fractals. */

#include "../include/fractal_color.h"
#include <string.h> /* memcpy */


/**
 * @brief create a mandelbrot mosaic made of julia fractals
 * 
 * @param savename the name to save to png image as
 * @param color the color mode for the image
 * @param fractal the fractal function to use
 * @param height_element the height of a single julia fractal (pixels)
 * @param width_element the width of a single julia fractal (pixels)
 * @param n_height the number of julia fractals to stack on top of each other
 * @param n_width the number of julia fractals to stack next to each other
 * @param c_real_start the start of the real part of the mandelbrot set to draw
 * @param c_real_end the end of the real part of the mandelbrot set to draw
 * @param c_imag_start the start of the imaginary part of the mandelbrot set to draw
 * @param c_imag_end the end of the imaginary part of the mandelbrot set to draw
 */
static void
julia_mandelbrot_photomosaic(
    char * savename, enum FC_Color color, enum FC_Fractal fractal,
    int height_element, int width_element,
    int n_height, int n_width,
    float c_real_start, float c_real_end, float c_imag_start, float c_imag_end)
{
    HCMATRIX hmosaic = fractal_cmatrix_create(n_height*height_element, n_width*width_element);
    HCMATRIX helement = fractal_cmatrix_create(height_element, width_element);

    float R = 4; // choose R outside of the visible range in the complex plane to avoid black outer edges in the julia fractals
    
    float aspect_ratio = (float)width_element/height_element;

    float x_start = -2;
    float x_end   = 2;

    float y_start = x_start / aspect_ratio;
    float y_end = x_end / aspect_ratio;

    struct FractalProperties fp = {
        .x_start = x_start,
        .x_end = x_end,
        .y_start = y_start,
        .y_end = y_end,
        .frac = fractal,
        .mode = FC_MODE_JULIA,
        .R = R,
        .max_iterations = 500,
    };

    for (int i=0; i<n_height; ++i) {
        fp.c_imag = c_imag_start + (i+0.5f)*(c_imag_end - c_imag_start)/n_height;
        for (int j=0; j<n_width; ++j) {
            fp.c_real = c_real_start + (j+0.5f)*(c_real_end - c_real_start)/n_width;
            
            fractal_avxf_get_colors(helement, &fp);

            // copy the julia fractal into the mosaic
            for (int h=i*height_element; h<(i+1)*height_element; ++h) {
                float * start_mosaic = fractal_cmatrix_value(hmosaic, h, j*width_element);
                float * start_element = fractal_cmatrix_value(helement, h-i*height_element, 0);
                memcpy(start_mosaic, start_element, sizeof(float)*width_element);
            }
        }
    }

    fractal_cmatrix_save(hmosaic, savename, color);

    fractal_cmatrix_free(hmosaic);
    fractal_cmatrix_free(helement);
}

int
main(void)
{
    julia_mandelbrot_photomosaic(
        "mosaic.png", FC_COLOR_ULTRA, FC_FRAC_Z2,
        40, 40,                  // julia fractal dimension (multiple of 8 for avx)
        200, 200,                // number of julia fractals that will be placed into a mosaic
        -2.0f, 0.8f, -1.4f, 1.4f // range in the complex plane to draw the mandelbrot set
    );

    return 0;
}
