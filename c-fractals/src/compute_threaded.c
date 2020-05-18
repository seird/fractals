#include "main.h"


static void *
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
