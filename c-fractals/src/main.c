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

    float x_start = -R;
    float x_end = R;

    float y_start = -R;
    float y_end = R; 

    int height = 16*100;
    int width = 16*100;

    int max_iterations = 1000;

    enum Mode mode = MODE_JULIA;
    enum Fractal fractal = FRAC_Z2;
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
    //fractal_get_colors_th(hCmatrix, &fp, 6);
    //fractal_avxf_get_colors(hCmatrix, &fp);
    fractal_avxf_get_colors_th(hCmatrix, &fp, 6);

    //float max_color = fractal_cmatrix_max(hCmatrix);

    fractal_cmatrix_save(hCmatrix, "fractal.png", COLOR_ULTRA);

    fractal_cmatrix_free(hCmatrix);

	return 0;
}
#endif
