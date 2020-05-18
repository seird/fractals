#include "compute_avx.h"


static void *
fractal_avxf_get_colors_thread_worker(void * arg)
{
    struct ThreadArg * targ = (struct ThreadArg *) arg;
    HS_CMATRIX hc = targ->hc;
    struct FractalProperties * fp = targ->fp;

    __m256 RR = _mm256_set1_ps(fp->R*fp->R);
    __m256 c_real = _mm256_set1_ps(fp->c_real);
    __m256 c_imag = _mm256_set1_ps(fp->c_imag);

    float x_step = fp->y_step;
    FRACDTYPE y = fp->y_start;
    for (int row=targ->row_start; row<targ->row_end; ++row) {
        FRACDTYPE x = fp->x_start;
        for (int col=0; col<hc->COLS; col+=VECFSIZE) {
            __m256 y_vec = _mm256_set1_ps(y);
            __m256 x_vec = _mm256_add_ps(
                _mm256_set1_ps(x),
                _mm256_set_ps(7*x_step, 6*x_step, 5*x_step, 4*x_step, 3*x_step, 2*x_step, x_step, 0) // Little endian
                //_mm256_set_ps(0, x_step, 2*x_step, 3*x_step, 4*x_step, 5*x_step, 6*x_step, 7*x_step)
            );

            fractal_avxf_get_vector_color(&hc->cmatrix[row][col], 
                                          &x_vec, &y_vec,
                                          &c_real, &c_imag,
                                          &RR, fp->max_iterations);

            x += VECFSIZE * x_step;
        }
        y += fp->y_step;
    }
    return NULL;
}

void
fractal_avxf_get_colors_th(HCMATRIX hCmatrix, struct FractalProperties * fp, int num_threads)
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

        if (pthread_create(&threads[i], NULL, fractal_avxf_get_colors_thread_worker, &args[i]) != 0) {
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
