#include "benchmarks.h"


#ifdef __AVX512DQ__

BENCH_FUNC(bench_avx512) {
    float c_real = C_REAL;
	float c_imag = C_IMAG;

    float _Complex c = c_real + c_imag*I;
    float R = ceilf(cabs(c)) + 1;

    HCMATRIX hCmatrix = fractal_cmatrix_create(HEIGHT, WIDTH);

    float x_start = -R;
    float x_end = R;

    float y_start = -R;
    float y_end = R;
    
    struct FractalProperties fp = {
        .x_start = x_start,
        .x_end = x_end,
        .y_start = y_start,
        .y_end = y_end,
        .frac = FRACTAL,
        .mode = MODE,
        .c_real = c_real,
        .c_imag = c_imag,
        .R = R,
        .max_iterations = MAX_ITERATIONS,
    };

    fractal_avx512f_get_colors(hCmatrix, &fp);

    fractal_cmatrix_free(hCmatrix);
}

#endif // __AVX512DQ__
