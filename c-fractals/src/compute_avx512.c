#include "compute_avx.h"


#ifdef __AVX512DQ__


__mmask16
fractal_avx512f_escape_magnitude_check(__m512 * z_real, __m512 * z_imag,
                                       __m512 * RR)
{
    // z_real * z_real + z_imag * z_imag > R*R

    // z_real * z_real
    __m512 z_real_sq = _mm512_mul_ps(*z_real, *z_real);

    // z_imag * z_imag
    __m512 z_imag_sq = _mm512_mul_ps(*z_imag, *z_imag);

    // z_real * z_real + z_imag * z_imag
    __m512 z_sq_sum = _mm512_add_ps(z_real_sq, z_imag_sq);

    // z_real * z_real + z_imag * z_imag > R*R
    __mmask16 escaped_mask = _mm512_cmp_ps_mask(z_sq_sum, *RR, _CMP_GT_OQ); // dst[i+31:i] := ( a[i+31:i] OP b[i+31:i] ) ? 1 : 0
    return escaped_mask;
}

void
fractal_avx512f_get_vector_color(float * color_array, 
                                 __m512 * z_real, __m512 * z_imag,
                                 __m512 * c_real, __m512 * c_imag,
                                 __m512 * RR, int max_iterations,
                                 fractal_avx512_t fractal)
    {
    __m512 colors_vec = _mm512_set1_ps(0);
    __mmask16 escaped_so_far_mask = 0;

	for (int iteration = 0 ; iteration < max_iterations; ++iteration) {
        // get all pixels that escaped this iteration
		__mmask16 escaped_mask = fractal_avx512f_escape_magnitude_check(z_real, z_imag, RR);

        // get pixels that escaped for the first time
        __mmask16 escaped_this_iteration_mask = escaped_mask & (escaped_mask ^ escaped_so_far_mask);

        // for the newly escaped pixels, set the escape iteration
        // color is iteration if mask = 1, else the color value remains the same 
        colors_vec = _mm512_mask_blend_ps(escaped_this_iteration_mask, colors_vec, _mm512_set1_ps(iteration));

        // update pixels that escaped this iteration
        escaped_so_far_mask = escaped_so_far_mask ^ escaped_this_iteration_mask;

        // break if all pixels have escaped
        if (escaped_so_far_mask == 0xFFFF) break; // mask == 0b1111111111111111 --> all 16 pixels escaped

        // next fractal step
		fractal(z_real, z_imag, z_real, z_imag, c_real, c_imag);
	}

    // store the pixel values
    _mm512_store_ps(color_array, colors_vec);
}

void
fractal_avx512f_get_colors(HCMATRIX hCmatrix, struct FractalProperties * fp)
{
    HS_CMATRIX hc = (HS_CMATRIX) hCmatrix;

    if (fp->mode != FC_MODE_FLAMES) {
        fp->_x_step = (fp->x_end - fp->x_start) / hc->width;
        fp->_y_step = (fp->y_end - fp->y_start) / hc->height;
    }
    
    fractal_avx512_t fractal = fractal_avx512_get(fp->frac);

    switch (fp->mode)
    {
        case FC_MODE_JULIA:
        {
            __m512 RR = _mm512_set1_ps(fp->R*fp->R);
            __m512 c_real = _mm512_set1_ps(fp->c_real);
            __m512 c_imag = _mm512_set1_ps(fp->c_imag);

            float x_step = fp->_x_step;
            float y = fp->y_start;
            for (int h=0; h<hc->height; ++h) {
                float x = fp->x_start;
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
                y += fp->_y_step;
            }
            break;
        }
        case FC_MODE_MANDELBROT:
        {
            __m512 RR = _mm512_set1_ps(fp->R*fp->R);

            float x_step = fp->_y_step;
            float y = fp->y_start;
            for (int h=0; h<hc->height; ++h) {
                float x = fp->x_start;
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
                        //_mm512_set_ps(0, x_step, 2*x_step, 3*x_step, 4*x_step, 5*x_step, 6*x_step, 7*x_step)
                    );

                    fractal_avx512f_get_vector_color(&hc->cmatrix[h][w], 
                                                &z_real, &z_imag,
                                                &c_real, &c_imag,
                                                &RR, fp->max_iterations,
                                                fractal);

                    x += VEC512FSIZE * x_step;
                }
                y += fp->_y_step;
            }
            break;
        }
        default: printf("Unsupported mode.\n");
    }
}

#endif // __AVX512DQ__
