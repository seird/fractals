#include "benchmarks.h"


BENCH_FUNC(bench_threaded) {
    float c_real = C_REAL;
	float c_imag = C_IMAG;

    float _Complex c = c_real + c_imag*I;
    float R = ceilf(cabs(c)) + 1;

    HCMATRIX hCmatrix = fractal_cmatrix_create(ROWS, COLS);

    float x_start = -R;
    float x_end = R;

    float y_start = -R;
    float y_end = R;
    
    struct FractalProperties fp = {
        .x_start = x_start,
        .x_end = x_end,
        .y_start = y_start,
        .y_end = y_end,
        .height = ROWS,
        .width = COLS,
        .frac = FRACTAL,
        .mode = MODE,
        .c_real = c_real,
        .c_imag = c_imag,
        .R = R,
        .max_iterations = MAX_ITERATIONS,
    };

    fractal_get_colors_th(hCmatrix, &fp, NUM_THREADS);
    
    fractal_cmatrix_free(hCmatrix);
}
