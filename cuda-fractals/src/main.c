#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "../include/fractal_cuda.h"


#define WIDTH   80 // 8*240
#define HEIGHT  40 //  10*108


int
main(int argc, char * argv[])
{    
    printf("Hello world\n");

    uint8_t * image = (uint8_t *) malloc(sizeof(int) * WIDTH*HEIGHT);
    float c_real = -0.788485f;
	float c_imag = 0.004913f;

    float R = ceilf(sqrtf(c_real*c_real+c_imag*c_imag)) + 1;

    float aspect_ratio = (float)WIDTH/HEIGHT;

    float x_start = 0.0f; // -R/5+0.5;
    float x_end   = 4.0f; //  R/5+0.5;

    float y_start = 0.0f; // x_start/aspect_ratio;
    float y_end = 4.0f; // x_end/aspect_ratio;

    struct FractalProperties fp = {
        .x_start = x_start,
        .x_end = x_end,
        .y_start = y_start,
        .y_end = y_end,
        .height = HEIGHT,
        .width = WIDTH,
        .frac = FC_FRAC_Z4,
        .mode = FC_MODE_LYAPUNOV, // TODO
        .c_real = c_real,
        .c_imag = c_imag,
        .R = R,
        .max_iterations = 1000,
        .sequence = "AABAB",
        .sequence_length = 5,
    };

    fractal_cuda_init(WIDTH, HEIGHT);
    for (int i=0; i<1; ++i) {
        fractal_cuda_get_colors(image, &fp);
    }
    fractal_cuda_clean();

    for (int h=0; h<HEIGHT; ++h) {
        for (int w=0; w<WIDTH; ++w) {
            // printf("%2d ", image[h*WIDTH+w]);
            printf(image[h*WIDTH+w] > 0 ? "*" : ".");
        }
        putchar('\n');
    }

    free(image);

    printf("Bye world\n");

    return 0;
}
