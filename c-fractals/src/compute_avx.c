#include "compute_avx.h"


void
fractal_avxf_print_vec(__m256 * v)
{
    for (float * f = (float*)v; f < (float*)v + VECFSIZE; ++f) {
        printf("%5f ", *f);
    }
    putchar('\n');
}

void
fractal_avxf_escape_magnitude_check(__m256 * escaped_mask,
                                    __m256 * z_real, __m256 * z_imag,
                                    __m256 * RR)
{
    // z_real * z_real + z_imag * z_imag > R*R

    // z_real * z_real
    __m256 z_real_sq = _mm256_mul_ps(*z_real, *z_real);

    // z_imag * z_imag
    __m256 z_imag_sq = _mm256_mul_ps(*z_imag, *z_imag);

    // z_real * z_real + z_imag * z_imag
    __m256 z_sq_sum = _mm256_add_ps(z_real_sq, z_imag_sq);

    // z_real * z_real + z_imag * z_imag > R*R
    *escaped_mask = _mm256_cmp_ps(z_sq_sum, *RR, _CMP_GT_OQ); // dst[i+31:i] := ( a[i+31:i] OP b[i+31:i] ) ? 0xFFFFFFFF : 0
}

void
fractal_avxf_get_vector_color(float * color_array, 
                              __m256 * z_real, __m256 * z_imag,
                              __m256 * c_real, __m256 * c_imag,
                              __m256 * RR, int max_iterations,
                              fractal_avx_t fractal)
{
    __m256 colors_vec = _mm256_set1_ps(0);
    __m256 escaped_so_far_mask = _mm256_set1_ps(0);
    __m256 escaped_mask;

	for (int iteration = 0 ; iteration < max_iterations; ++iteration) {
        // get all pixels that escaped this iteration
		fractal_avxf_escape_magnitude_check(&escaped_mask, z_real, z_imag, RR);

        // get pixels that escaped for the first time
        __m256 escaped_this_iteration_mask = _mm256_and_ps(
            escaped_mask,
            _mm256_xor_ps(escaped_mask, escaped_so_far_mask)
        );

        // for the newly escaped pixels, set the escape iteration
        // color is iteration if mask = 1, else the color value remains the same 
        colors_vec =_mm256_blendv_ps(colors_vec, _mm256_set1_ps(iteration), escaped_this_iteration_mask);

        // update pixels that escaped this iteration
        escaped_so_far_mask = _mm256_or_ps(escaped_so_far_mask, escaped_this_iteration_mask);

        // break if all pixels have escaped
        if (_mm256_movemask_ps(escaped_so_far_mask) == 255) break; // mask == 0b11111111

        // next fractal step
		fractal(z_real, z_imag, z_real, z_imag, c_real, c_imag);
	}

    // store the pixel values
    _mm256_store_ps(color_array, colors_vec);
}

void
fractal_avxf_get_colors(HCMATRIX hCmatrix, struct FractalProperties * fp)
{
    HS_CMATRIX hc = (HS_CMATRIX) hCmatrix;

    fp->_x_step = (fp->x_end - fp->x_start) / fp->width;
    fp->_y_step = (fp->y_end - fp->y_start) / fp->height;

    fractal_avx_t fractal = fractal_avx_get(fp->frac);

    switch (fp->mode)
    {
        case MODE_JULIA:
        {
            __m256 RR = _mm256_set1_ps(fp->R*fp->R);
            __m256 c_real = _mm256_set1_ps(fp->c_real);
            __m256 c_imag = _mm256_set1_ps(fp->c_imag);

            float x_step = fp->_x_step;
            float y = fp->y_start;
            for (int row=0; row<hc->ROWS; ++row) {
                float x = fp->x_start;
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
                                                  &RR, fp->max_iterations,
                                                  fractal);

                    x += VECFSIZE * x_step;
                }
                y += fp->_y_step;
            }
            break;
        }
        case MODE_MANDELBROT:
        {
            __m256 RR = _mm256_set1_ps(fp->R*fp->R);

            float x_step = fp->_y_step;
            float y = fp->y_start;
            for (int row=0; row<hc->ROWS; ++row) {
                float x = fp->x_start;
                for (int col=0; col<hc->COLS; col+=VECFSIZE) {
                    // Start at z=0
                    __m256 z_real = _mm256_set1_ps(0);
                    __m256 z_imag = _mm256_set1_ps(0);

                    __m256 c_real = _mm256_set1_ps(y);
                    __m256 c_imag = _mm256_add_ps(
                        _mm256_set1_ps(x),
                        _mm256_set_ps(7*x_step, 6*x_step, 5*x_step, 4*x_step, 3*x_step, 2*x_step, x_step, 0) // Little endian
                        //_mm256_set_ps(0, x_step, 2*x_step, 3*x_step, 4*x_step, 5*x_step, 6*x_step, 7*x_step)
                    );

                    fractal_avxf_get_vector_color(&hc->cmatrix[row][col], 
                                                &z_real, &z_imag,
                                                &c_real, &c_imag,
                                                &RR, fp->max_iterations,
                                                fractal);

                    x += VECFSIZE * x_step;
                }
                y += fp->_y_step;
            }
            break;
        }
        default: printf("Unsupported mode.\n");
    }
}
