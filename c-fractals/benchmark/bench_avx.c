#include "benchmarks.h"


BENCH_FUNC(bench_avx) {
    FRACDTYPE c_real = C_REAL;
	FRACDTYPE c_imag = C_IMAG;

    FRACDTYPE _Complex c = c_real + c_imag*I;
    FRACDTYPE R = ceilf(cabs(c)) + 1;

    HCMATRIX hCmatrix = fractal_cmatrix_create(ROWS, COLS);

    FRACDTYPE x_start = -R;
    FRACDTYPE x_end = R;

    FRACDTYPE y_start = -R;
    FRACDTYPE y_end = R;

    FRACDTYPE x_step = (x_end - x_start) / ROWS;
    FRACDTYPE y_step = (y_end - y_start) / COLS;
    
    struct FractalProperties fp = {
        .x_start = -R,
        .x_step = x_step,
        .y_start = y_start,
        .y_step = y_step,
        .frac = FRACTAL,
        .mode = MODE,
        .c_real = c_real,
        .c_imag = c_imag,
        .R = R,
        .max_iterations = MAX_ITERATIONS,
    };

    fractal_avxf_get_colors(hCmatrix, &fp);

    fractal_cmatrix_free(hCmatrix);
}
