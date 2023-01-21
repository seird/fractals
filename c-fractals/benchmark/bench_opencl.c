#include "benchmarks.h"


#ifdef OPENCL

BENCH_FUNC(bench_opencl) {
    float c_real = C_REAL;
	float c_imag = C_IMAG;

    float _Complex c = c_real + c_imag*I;
    float R = ceilf(cabs(c)) + 1;

    uint8_t * image = (uint8_t *) malloc(sizeof(uint8_t)*HEIGHT*WIDTH*3);


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

    fractal_opencl_get_colors(image, &fp);
    free(image);
}

#endif // OPENCL
