#include "main.h"


bool
escape_magnitude_check(float _Complex z, float R)
{
	return (crealf(z) * crealf(z) + cimagf(z) * cimagf(z)) > (R * R);
}

void
fractal_get_single_color(float * color, float x, float y, float _Complex (*fractal)(float complex, float _Complex), float _Complex c, float R, int max_iterations)
{
	int num_iterations = 0;
	float _Complex z = x + y*I;

	for (; num_iterations < max_iterations; ++num_iterations) {
		if (escape_magnitude_check(z, R))
			break;
		z = (*fractal)(z, c);
	}

	*color = num_iterations == max_iterations ? BLACK : num_iterations;
}

void
fractal_get_colors_cmpx(
    HCMATRIX hCmatrix,
    float x_start, float x_step, float y_start, float y_step,
    enum Fractal frac, float _Complex c,
    float R, int max_iterations)
{
    HS_CMATRIX hc = (HS_CMATRIX) hCmatrix;

    float _Complex (*fractal)(float complex, float _Complex) = fractal_get(frac);

    float x = x_start;
    for (int row=0; row<hc->ROWS; ++row) {
        float y = y_start;
        for (int col=0; col<hc->COLS; ++col) {
            fractal_get_single_color(&hc->cmatrix[row][col], x, y, fractal, c, R, max_iterations);
            y += y_step;
        }
        x += x_step;
    }
}

void
fractal_get_colors(
    HCMATRIX hCmatrix,
    float x_start, float x_step, float y_start, float y_step,
    enum Fractal frac, float c_real, float c_imag,
    float R, int max_iterations)
{
    HS_CMATRIX hc = (HS_CMATRIX) hCmatrix;

    float _Complex c = c_real + c_imag * I;

    float _Complex (*fractal)(float complex, float _Complex) = fractal_get(frac);

    float x = x_start;
    for (int row=0; row<hc->ROWS; ++row) {
        float y = y_start;
        for (int col=0; col<hc->COLS; ++col) {
            fractal_get_single_color(&hc->cmatrix[row][col], x, y, fractal, c, R, max_iterations);
            y += y_step;
        }
        x += x_step;
    }
}

void *
get_colors_thread_worker(void * arg)
{
    struct ThreadArg * targ = (struct ThreadArg *) arg;
    HS_CMATRIX hc = targ->hc;

    float _Complex c = targ->c_real + targ->c_imag * I;
    float _Complex (*fractal)(float complex, float _Complex) = fractal_get(targ->frac);

    float x = targ->x_start;
    for (int row=targ->row_start; row<targ->row_end; ++row) {
        float y = targ->y_start;
        for (int col=0; col<hc->COLS; ++col) {
            fractal_get_single_color(&hc->cmatrix[row][col], x, y, fractal, c, targ->R, targ->max_iterations);
            y += targ->y_step;
        }
        x += targ->x_step;
    }
    return NULL;
}

void
fractal_get_colors_th(
    HCMATRIX hCmatrix,
    float x_start, float x_step, float y_start, float y_step,
    enum Fractal frac, float c_real, float c_imag,
    float R, int max_iterations, int num_threads)
{
    HS_CMATRIX hc = (HS_CMATRIX) hCmatrix;

    pthread_t threads[num_threads];
    struct ThreadArg args[num_threads];
    for (int i=0; i<num_threads; ++i) {
        args[i].hc = hc;
        args[i].row_start = i*hc->ROWS/num_threads;
        args[i].row_end = (i+1)*hc->ROWS/num_threads;
        args[i].x_start = x_start + x_step*i*(float)hc->ROWS/num_threads;
        args[i].x_step = x_step;
        args[i].y_start = y_start;
        args[i].y_step = y_step;
        args[i].frac = frac;
        args[i].c_real = c_real;
        args[i].c_imag = c_imag;
        args[i].R = R;
        args[i].max_iterations = max_iterations;

        if (pthread_create(&threads[i], NULL, get_colors_thread_worker, &args[i]) != 0) {
            printf("Thread %d could not be created.\n", i);
        }
    }

    for (int i=0; i<num_threads; ++i) {
        if (pthread_join(threads[i], NULL) != 0) {
            printf("Thread %d could not be joined.\n", i);
        }

    }
}

float
fractal_get_max_color(HCMATRIX hCmatrix)
{
    HS_CMATRIX hc = (HS_CMATRIX) hCmatrix;

    float max_color = 0.0;
    for (int row=0; row<hc->ROWS; ++row) {
        for (int col=0; col<hc->COLS; ++col) {
            if (hc->cmatrix[row][col] > max_color) {
                max_color = hc->cmatrix[row][col];
            }
        }
    }
    return max_color;
}