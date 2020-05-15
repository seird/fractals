#include "main.h"


bool
escape_magnitude_check(FRACDTYPE _Complex z, FRACDTYPE R)
{
	return (crealf(z) * crealf(z) + cimagf(z) * cimagf(z)) > (R * R);
}

void
fractal_get_single_color(FRACDTYPE * color, FRACDTYPE x, FRACDTYPE y, FRACDTYPE _Complex (*fractal)(FRACDTYPE complex, FRACDTYPE _Complex), FRACDTYPE _Complex c, FRACDTYPE R, int max_iterations)
{
	int num_iterations = 0;
	FRACDTYPE _Complex z = x + y*I;

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

    FRACDTYPE _Complex c = fp->c_real + fp->c_imag * I;
    FRACDTYPE _Complex (*fractal)(FRACDTYPE complex, FRACDTYPE _Complex) = fractal_get(fp->frac);

    FRACDTYPE y = fp->y_start;
    for (int row=0; row<hc->ROWS; ++row) {
        FRACDTYPE x = fp->x_start;
        for (int col=0; col<hc->COLS; ++col) {
            fractal_get_single_color(&hc->cmatrix[row][col], x, y, fractal, c, fp->R, fp->max_iterations);
            x += fp->x_step;
        }
        y += fp->y_step;
    }
}

void *
get_colors_thread_worker(void * arg)
{
    struct ThreadArg * targ = (struct ThreadArg *) arg;
    HS_CMATRIX hc = targ->hc;
    struct FractalProperties * fp = targ->fp;

    FRACDTYPE _Complex c = fp->c_real + fp->c_imag * I;
    FRACDTYPE _Complex (*fractal)(FRACDTYPE complex, FRACDTYPE _Complex) = fractal_get(fp->frac);

    FRACDTYPE y = fp->y_start;
    for (int row=targ->row_start; row<targ->row_end; ++row) {
        FRACDTYPE x = fp->x_start;
        for (int col=0; col<hc->COLS; ++col) {
            fractal_get_single_color(&hc->cmatrix[row][col], x, y, fractal, c, fp->R, fp->max_iterations);
            x += fp->x_step;
        }
        y += fp->y_step;
    }
    return NULL;
}

void
fractal_get_colors_th(HCMATRIX hCmatrix, struct FractalProperties * fp, int num_threads)
{
    HS_CMATRIX hc = (HS_CMATRIX) hCmatrix;

    pthread_t threads[num_threads];
    struct ThreadArg args[num_threads];
    for (int i=0; i<num_threads; ++i) {
        args[i].hc = hc;
        args[i].row_start = i*hc->ROWS/num_threads;
        args[i].row_end = (i+1)*hc->ROWS/num_threads;
        args[i].fp = malloc(sizeof(struct FractalProperties));
        memcpy(args[i].fp, fp, sizeof(struct FractalProperties));
        args[i].fp->y_start = fp->y_start + fp->y_step*i*(FRACDTYPE)hc->ROWS/num_threads;

        if (pthread_create(&threads[i], NULL, get_colors_thread_worker, &args[i]) != 0) {
            printf("Thread %d could not be created.\n", i);
        }
    }

    for (int i=0; i<num_threads; ++i) {
        if (pthread_join(threads[i], NULL) != 0) {
            printf("Thread %d could not be joined.\n", i);
        }
        free(args[i].fp);

    }
}

FRACDTYPE
fractal_get_max_color(HCMATRIX hCmatrix)
{
    HS_CMATRIX hc = (HS_CMATRIX) hCmatrix;

    FRACDTYPE max_color = 0.0;
    for (int row=0; row<hc->ROWS; ++row) {
        for (int col=0; col<hc->COLS; ++col) {
            if (hc->cmatrix[row][col] > max_color) {
                max_color = hc->cmatrix[row][col];
            }
        }
    }
    return max_color;
}