#include "main.h"


bool
escape_magnitude_check(double _Complex z, double R)
{
	return (crealf(z) * crealf(z) + cimagf(z) * cimagf(z)) > (R * R);
}

void
fractal_get_single_color(double * color, double x, double y, double _Complex (*fractal)(double complex, double _Complex), double _Complex c, double R, int max_iterations)
{
	int num_iterations = 0;
	double _Complex z = x + y*I;

	for (; num_iterations < max_iterations; ++num_iterations) {
		if (escape_magnitude_check(z, R))
			break;
		z = (*fractal)(z, c);
	}

	*color = num_iterations == max_iterations ? BLACK : num_iterations;
}

void
fractal_get_colors(HCMATRIX hCmatrix, struct FractalProperties * fp)
{
    HS_CMATRIX hc = (HS_CMATRIX) hCmatrix;

    double _Complex c = fp->c_real + fp->c_imag * I;

    double _Complex (*fractal)(double complex, double _Complex) = fractal_get(fp->frac);

    double x = fp->x_start;
    for (int row=0; row<hc->ROWS; ++row) {
        double y = fp->y_start;
        for (int col=0; col<hc->COLS; ++col) {
            fractal_get_single_color(&hc->cmatrix[row][col], x, y, fractal, c, fp->R, fp->max_iterations);
            y += fp->y_step;
        }
        x += fp->x_step;
    }
}

void *
get_colors_thread_worker(void * arg)
{
    struct ThreadArg * targ = (struct ThreadArg *) arg;
    HS_CMATRIX hc = targ->hc;
    struct FractalProperties * fp = targ->fp;

    double _Complex c = fp->c_real + fp->c_imag * I;
    double _Complex (*fractal)(double complex, double _Complex) = fractal_get(fp->frac);

    double x = fp->x_start;
    for (int row=targ->row_start; row<targ->row_end; ++row) {
        double y = fp->y_start;
        for (int col=0; col<hc->COLS; ++col) {
            fractal_get_single_color(&hc->cmatrix[row][col], x, y, fractal, c, fp->R, fp->max_iterations);
            y += fp->y_step;
        }
        x += fp->x_step;
    }
    return NULL;
}

void
fractal_get_colors_th(
    HCMATRIX hCmatrix, struct FractalProperties * fp, int num_threads)
{
    HS_CMATRIX hc = (HS_CMATRIX) hCmatrix;

    pthread_t threads[num_threads];
    struct ThreadArg args[num_threads];
    struct FractalProperties props[num_threads];
    for (int i=0; i<num_threads; ++i) {
        args[i].hc = hc;
        args[i].row_start = i*hc->ROWS/num_threads;
        args[i].row_end = (i+1)*hc->ROWS/num_threads;
        props[i].x_start = fp->x_start + fp->x_step*i*(double)hc->ROWS/num_threads;
        props[i].x_step = fp->x_step;
        props[i].y_start = fp->y_start;
        props[i].y_step = fp->y_step;
        props[i].frac = fp->frac;
        props[i].c_real = fp->c_real;
        props[i].c_imag = fp->c_imag;
        props[i].R = fp->R;
        props[i].max_iterations = fp->max_iterations;
        args[i].fp = &props[i];

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

double
fractal_get_max_color(HCMATRIX hCmatrix)
{
    HS_CMATRIX hc = (HS_CMATRIX) hCmatrix;

    double max_color = 0.0;
    for (int row=0; row<hc->ROWS; ++row) {
        for (int col=0; col<hc->COLS; ++col) {
            if (hc->cmatrix[row][col] > max_color) {
                max_color = hc->cmatrix[row][col];
            }
        }
    }
    return max_color;
}