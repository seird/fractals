#include "tests.h"


MU_TEST(test_threaded_result)
{
    int num_threads = 2;

    FRACDTYPE c_real = -0.835f;
	FRACDTYPE c_imag = -0.2321f;

    FRACDTYPE _Complex c = c_real + c_imag*I;
    FRACDTYPE R = ceilf(cabs(c)) + 1;

    int ROWS = 100;
    int COLS = 100;

    FRACDTYPE x_start = -R;
    FRACDTYPE x_end = R;

    FRACDTYPE y_start = -R;
    FRACDTYPE y_end = R;

    FRACDTYPE x_step = (x_end - x_start) / ROWS;
    FRACDTYPE y_step = (y_end - y_start) / COLS;

    int max_iterations = 100;

    HCMATRIX hCmatrix_nc = fractal_cmatrix_create(ROWS, COLS);
    HCMATRIX hCmatrix_th = fractal_cmatrix_create(ROWS, COLS);

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

    fractal_get_colors(hCmatrix_nc, &fp);
    fractal_get_colors_th(hCmatrix_th, &fp, num_threads);

    FRACDTYPE max_color_nc = fractal_cmatrix_max(hCmatrix_nc);
    FRACDTYPE max_color_th = fractal_cmatrix_max(hCmatrix_th);

    MU_CHECK(max_color_nc == max_color_th);

    fractal_cmatrix_free(hCmatrix_nc);
    fractal_cmatrix_free(hCmatrix_th);
}
