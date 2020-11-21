#include "main.h"


#if (!defined(TEST) && !defined(SHARED) && !defined(BENCHMARK))
int
main(void)
{    
    /* ----------- INPUT PARAMETERS ----------- */
    float c_real = -0.7835f;
	float c_imag = -0.2321f;
    float _Complex c = c_real + c_imag*I;
    float R = ceilf(cabs(c)) + 1;
    
    int height = 10*108;
    int width = 8*240;
    
    float aspect_ratio = (float)width/height;

    float x_start = -R/5+0.5;
    float x_end   =  R/5+0.5;

    float y_start = x_start/aspect_ratio;
    float y_end = x_end/aspect_ratio;    

    int max_iterations = 1000;

    enum FC_Mode mode = FC_MODE_JULIA;
    enum FC_Fractal fractal = FC_FRAC_Z2;
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
        .c_real = c_real,
        .c_imag = c_imag,
        .R = R,
        .max_iterations = max_iterations,
    };

    HCMATRIX hCmatrix = fractal_cmatrix_create(height, width);

    //fractal_get_colors(hCmatrix, &fp);
    //fractal_get_colors_th(hCmatrix, &fp, 10);
    //fractal_avxf_get_colors(hCmatrix, &fp);
    fractal_avxf_get_colors_th(hCmatrix, &fp, 10);

    //float max_color = fractal_cmatrix_max(hCmatrix);

    fractal_cmatrix_save(hCmatrix, "fractal.png", FC_COLOR_MONOCHROME);

    fractal_cmatrix_free(hCmatrix);

	return 0;
}
#endif
