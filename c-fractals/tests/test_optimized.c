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

    float x_step = (x_end - x_start) / ROWS;
    float y_step = (y_end - y_start) / COLS;

    int max_iterations = 100;

    HCMATRIX hCmatrix = fractal_cmatrix_create(ROWS, COLS);
    HCMATRIX hCmatrix_nc = fractal_cmatrix_create(ROWS, COLS);
    HCMATRIX hCmatrix_th = fractal_cmatrix_create(ROWS, COLS);

    fractal_get_colors_cmpx(hCmatrix, x_start, x_step, y_start, y_step, FRAC_JULIA, c, R, max_iterations);
    fractal_get_colors(hCmatrix_nc, x_start, x_step, y_start, y_step, FRAC_JULIA, c_real, c_imag, R, max_iterations);
    fractal_get_colors_th(hCmatrix_th, x_start, x_step, y_start, y_step, FRAC_JULIA, c_real, c_imag, R, max_iterations, num_threads);

    float max_color = fractal_get_max_color(hCmatrix);
    float max_color_nc = fractal_get_max_color(hCmatrix_nc);
    float max_color_th = fractal_get_max_color(hCmatrix_th);

    MU_CHECK(max_color == max_color_nc);
    MU_CHECK(max_color == max_color_th);

    fractal_cmatrix_free(hCmatrix);
    fractal_cmatrix_free(hCmatrix_nc);
    fractal_cmatrix_free(hCmatrix_th);
}
