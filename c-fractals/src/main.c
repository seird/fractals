#include "main.h"


#if (!defined(TEST) && !defined(SHARED) && !defined(BENCHMARK))
int
main(void)
{    
    float c_real = -0.7835f;
	float c_imag = -0.2321f;

    float _Complex c = c_real + c_imag*I;
    float R = ceilf(cabs(c)) + 1;

    int ROWS = 16*100;
    int COLS = 16*100;

    HCMATRIX hCmatrix = fractal_cmatrix_create(1, 1);
    hCmatrix = fractal_cmatrix_reshape(hCmatrix, ROWS, COLS);

    float x_start = -R;
    float x_end = R;

    float y_start = -R;
    float y_end = R;

    float x_step = (x_end - x_start) / ROWS;
    float y_step = (y_end - y_start) / COLS;

    int max_iterations = 1000;
    
    struct FractalProperties fp = {
        .x_start = -R,
        .x_step = x_step,
        .y_start = y_start,
        .y_step = y_step,
        .frac = FRAC_Z2,
        .mode = MODE_JULIA,
        .c_real = c_real,
        .c_imag = c_imag,
        .R = R,
        .max_iterations = max_iterations,
    };

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
