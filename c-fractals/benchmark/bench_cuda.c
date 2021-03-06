#include "benchmarks.h"


#ifdef CUDA

BENCH_FUNC(bench_cuda) {
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

    fractal_cuda_get_colors(image, &fp);
    free(image);
}


BENCH_FUNC(bench_cuda_lyapunov) {
    uint8_t * image = (uint8_t *) malloc(sizeof(uint8_t)*HEIGHT*WIDTH*3);


    float x_start = 0;
    float x_end = 4;

    float y_start = 0;
    float y_end = 4;
    
    struct FractalProperties fp = {
        .x_start = x_start,
        .x_end = x_end,
        .y_start = y_start,
        .y_end = y_end,
        .mode = FC_MODE_LYAPUNOV,
        .sequence = "ABAAB",
        .sequence_length = 5,
        .max_iterations = MAX_ITERATIONS,
    };

    fractal_cuda_get_colors(image, &fp);
    free(image);
}


#endif // CUDA
