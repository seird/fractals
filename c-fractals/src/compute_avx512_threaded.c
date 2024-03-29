#include "compute_avx.h"

#ifdef __AVX512DQ__

static void *
fractal_avx512f_get_colors_thread_worker(void * arg)
{
    struct ThreadArg * targ = (struct ThreadArg *) arg;
    HS_CMATRIX hc = targ->hc;
    struct FractalProperties * fp = targ->fp;

    fractal_avx512_t fractal = fractal_avx512_get(fp->frac);

    switch (fp->mode)
    {
        case FC_MODE_JULIA:
        {
            __m512 RR = _mm512_set1_ps(fp->R*fp->R);
            __m512 c_real = _mm512_set1_ps(fp->c_real);
            __m512 c_imag = _mm512_set1_ps(fp->c_imag);

            float x_step = fp->_x_step;
            for (int h=targ->thread_id; h<hc->height; h+=targ->num_threads) {
                float x = fp->x_start;
                float y = fp->y_start + fp->_y_step*h;
                for (int w=0; w<hc->width; w+=VEC512FSIZE) {
                    __m512 y_vec = _mm512_set1_ps(y);
                    __m512 x_vec = _mm512_add_ps(
                        _mm512_set1_ps(x),
                        _mm512_set_ps(15*x_step, 14*x_step, 13*x_step, 12*x_step, 11*x_step, 10*x_step, 9*x_step, 8*x_step, 
                                      7*x_step,  6*x_step,  5*x_step,  4*x_step,  3*x_step,  2*x_step,  x_step,   0)
                                      // Little endian
                    );

                    fractal_avx512f_get_vector_color(&hc->cmatrix[h][w], 
                                                     &x_vec, &y_vec,
                                                     &c_real, &c_imag,
                                                     &RR, fp->max_iterations,
                                                     fractal);

                    x += VEC512FSIZE * x_step;
                }
            }
            break;
        }
        case FC_MODE_MANDELBROT:
        {
            __m512 RR = _mm512_set1_ps(fp->R*fp->R);

            float x_step = fp->_x_step;
            for (int h=targ->thread_id; h<hc->height; h+=targ->num_threads) {
                float x = fp->x_start;
                float y = fp->y_start + fp->_y_step*h;
                for (int w=0; w<hc->width; w+=VEC512FSIZE) {
                    // Start at z=0
                    __m512 z_real = _mm512_set1_ps(0);
                    __m512 z_imag = _mm512_set1_ps(0);

                    __m512 c_real = _mm512_set1_ps(y);
                    __m512 c_imag = _mm512_add_ps(
                        _mm512_set1_ps(x),
                        _mm512_set_ps(15*x_step, 14*x_step, 13*x_step, 12*x_step, 11*x_step, 10*x_step, 9*x_step, 8*x_step, 
                                      7*x_step,  6*x_step,  5*x_step,  4*x_step,  3*x_step,  2*x_step,  x_step,   0)
                                      // Little endian
                    );

                    fractal_avx512f_get_vector_color(&hc->cmatrix[h][w], 
                                                &z_real, &z_imag,
                                                &c_real, &c_imag,
                                                &RR, fp->max_iterations,
                                                fractal);

                    x += VEC512FSIZE * x_step;
                }
            }
            break;
        }
        default: printf("Unsupported mode.\n");
    }
    return NULL;
}

void
fractal_avx512f_get_colors_th(HCMATRIX hCmatrix, struct FractalProperties * fp, int num_threads)
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

        if (pthread_create(&threads[i], NULL, fractal_avx512f_get_colors_thread_worker, &args[i]) != 0) {
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

#endif // __AVX512DQ__
