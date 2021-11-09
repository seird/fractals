#include "../include/fractal_cuda.h"


int
main(void)
{
    /* ----------- INPUT PARAMETERS ----------- */
    float c_real = -0.7835f;
	float c_imag = -0.2321f;
    float R = 2.0f;
    
    int height = 8*100;
    int width = 8*240;
    
    float aspect_ratio = (float)width/height;

    float x_start = -R;
    float x_end   =  R;

    float y_start = x_start/aspect_ratio;
    float y_end = x_end/aspect_ratio;    

    int max_iterations = 1000;

    enum FC_Mode mode = FC_MODE_JULIA;
    enum FC_Fractal fractal = FC_FRAC_Z2;
    enum FC_Color color = FC_COLOR_ULTRA;
    /* ---------------------------------------- */


    struct FractalProperties fp = {
        .x_start = x_start,
        .x_end = x_end,
        .y_start = y_start,
        .y_end = y_end,

        .width = width,
        .height = height,

        .frac = fractal,
        .mode = mode,
        .color = color,

        .c_real = c_real,
        .c_imag = c_imag,
        .R = R,
        .max_iterations = max_iterations,
    };

    fractal_cuda_init(width, height);
    uint8_t * image = fractal_cuda_image_create(width, height);

    fractal_cuda_get_colors(image, &fp);

    fractal_cuda_image_save(image, width, height, "julia_cuda.png");

    fractal_cuda_image_free(image);

    return 0;
}
