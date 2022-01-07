#include "main.h"


static void *
get_colors_thread_worker(void * arg)
{
    struct ThreadArg * targ = (struct ThreadArg *) arg;
    HS_CMATRIX hc = targ->hc;
    struct FractalProperties * fp = targ->fp;

    float _Complex c = fp->c_real + fp->c_imag * I;
    fractal_t fractal = fractal_get(fp->frac);

    for (int h=targ->thread_id; h<hc->height; h+=targ->num_threads) {
        float x = fp->x_start;
        float y = fp->y_start + fp->_y_step*h;
        for (int w=0; w<hc->width; ++w) {
            fractal_get_single_color(&hc->cmatrix[h][w], x, y, fractal, c, fp->R, fp->max_iterations);
            x += fp->_x_step;
        }
    }
    return NULL;
}

void
fractal_get_colors_th(HCMATRIX hCmatrix, struct FractalProperties * fp, int num_threads)
{
    HS_CMATRIX hc = (HS_CMATRIX) hCmatrix;

    if (fp->mode != FC_MODE_FLAMES) {
        fp->_x_step = (fp->x_end - fp->x_start) / hc->width;
        fp->_y_step = (fp->y_end - fp->y_start) / hc->height;
    }

    pthread_t threads[num_threads];
    struct ThreadArg args[num_threads];
    for (int i=0; i<num_threads; ++i) {
        args[i].hc = hc;
        args[i].thread_id = i;
        args[i].num_threads = num_threads;
        args[i].fp = malloc(sizeof(struct FractalProperties));
        memcpy(args[i].fp, fp, sizeof(struct FractalProperties));
        // args[i].fp->y_start = fp->y_start + fp->_y_step*i*(float)hc->height/num_threads;

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
