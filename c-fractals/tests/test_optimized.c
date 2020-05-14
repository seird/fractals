#include "tests.h"


MU_TEST(test_threaded_result)
{
    int num_threads = 2;

    double c_real = -0.835f;
	double c_imag = -0.2321f;

    double _Complex c = c_real + c_imag*I;
    double R = ceilf(cabs(c)) + 1;

    int ROWS = 100;
    int COLS = 100;

    double x_start = -R;
    double x_end = R;

    double y_start = -R;
    double y_end = R;

    double x_step = (x_end - x_start) / ROWS;
    double y_step = (y_end - y_start) / COLS;

    int max_iterations = 100;

    HCMATRIX hCmatrix_nc = fractal_cmatrix_create(ROWS, COLS);
    HCMATRIX hCmatrix_th = fractal_cmatrix_create(ROWS, COLS);

    struct FractalProperties fp = {
        .x_start = -R,
        .x_step = x_step,
        .y_start = y_start,
        .y_step = y_step,
        .frac = FRAC_JULIA,
        .c_real = c_real,
        .c_imag = c_imag,
        .R = R,
        .max_iterations = max_iterations,
    };

    fractal_get_colors(hCmatrix_nc, &fp);
    fractal_get_colors_th(hCmatrix_th, &fp, num_threads);

    double max_color_nc = fractal_get_max_color(hCmatrix_nc);
    double max_color_th = fractal_get_max_color(hCmatrix_th);

    MU_CHECK(max_color_nc == max_color_th);

    fractal_cmatrix_free(hCmatrix_nc);
    fractal_cmatrix_free(hCmatrix_th);
}
