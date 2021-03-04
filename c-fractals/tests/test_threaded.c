#include "tests.h"


MU_TEST(test_threaded_result)
{
    int num_threads = 2;

    float c_real = -0.835f;
	float c_imag = -0.2321f;

    float _Complex c = c_real + c_imag*I;
    float R = ceilf(cabs(c)) + 1;

    int ROWS = 100;
    int COLS = 100;

    float x_start = -R;
    float x_end = R;

    float y_start = -R;
    float y_end = R;

    int max_iterations = 100;

    HCMATRIX hCmatrix_nc = fractal_cmatrix_create(ROWS, COLS);
    HCMATRIX hCmatrix_th = fractal_cmatrix_create(ROWS, COLS);

    struct FractalProperties fp = {
        .x_start = x_start,
        .x_end = x_end,
        .y_start = y_start,
        .y_end = y_end,
        .height = ROWS,
        .width = COLS,
        .frac = FC_FRAC_Z2,
        .mode = FC_MODE_JULIA,
        .c_real = c_real,
        .c_imag = c_imag,
        .R = R,
        .max_iterations = max_iterations,
    };

    fractal_get_colors(hCmatrix_nc, &fp);
    fractal_get_colors_th(hCmatrix_th, &fp, num_threads);

    float max_color_nc = fractal_cmatrix_max(hCmatrix_nc);
    float max_color_th = fractal_cmatrix_max(hCmatrix_th);

    MU_CHECK_FLT_EQ(max_color_nc, max_color_th);

    fractal_cmatrix_free(hCmatrix_nc);
    fractal_cmatrix_free(hCmatrix_th);
}
