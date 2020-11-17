#include "benchmarks.h"


BENCH_FUNC(bench_avx) {
    float c_real = C_REAL;
	float c_imag = C_IMAG;

    float _Complex c = c_real + c_imag*I;
    float R = ceilf(cabs(c)) + 1;

    HCMATRIX hCmatrix = fractal_cmatrix_create(ROWS, COLS);

    float x_start = -R;
    float x_end = R;

    float y_start = -R;
    float y_end = R;

    float x_step = (x_end - x_start) / ROWS;
    float y_step = (y_end - y_start) / COLS;
    
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
