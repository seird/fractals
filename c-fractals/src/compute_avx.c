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
fractal_avxf_julia(__m256 * result_real, __m256 * result_imag, 
                   __m256 * z_real, __m256 * z_imag, 
                   __m256 * c_real, __m256 * c_imag)
{
    // z * z = (a+bj)*(a+bj) = a*a - b*b + 2(a*b)j

    // z_real*z_real
    __m256 z_real_sq = _mm256_mul_ps(*z_real, *z_real);

    // z_imag*z_imag
    __m256 z_imag_sq = _mm256_mul_ps(*z_imag, *z_imag);

    // 2*z_real*z_imag
    __m256 z_real_imag_prod = _mm256_mul_ps(
        _mm256_mul_ps(*z_real, *z_imag),
        _mm256_set1_ps(2)
    );

    // z_real*z_real - z_imag*z_imag
    __m256 z_sq_diff = _mm256_sub_ps(z_real_sq, z_imag_sq);

    // z_real*z_real + c_real
    *result_real = _mm256_add_ps(z_sq_diff, *c_real);

    // z_imag*z_imag + c_imag
    *result_imag = _mm256_add_ps(z_real_imag_prod, *c_imag);
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
fractal_avxf_get_vector_color(FRACDTYPE * color_array, 
                              __m256 * z_real, __m256 * z_imag,
                              __m256 * c_real, __m256 * c_imag,
                              __m256 * RR, int max_iterations)
{
	int iteration = 0;
    __m256 colors_vec = _mm256_set1_ps(0);
    __m256 escaped_mask, escaped_so_far_mask;

	for (; iteration < max_iterations; ++iteration) {
        // get all pixels that escaped this iteration
		fractal_avxf_escape_magnitude_check(&escaped_mask, z_real, z_imag, RR);

        // get pixels that escaped for the first time
        __m256 escaped_this_round_mask = _mm256_and_ps(
            escaped_mask,
            _mm256_xor_ps(escaped_mask, escaped_so_far_mask)
        );

        // for the newly escaped pixels, set the escape iteration
        //_mm256_maskstore_ps(color, _mm256_castps_si256(escaped_this_round_mask), _mm256_set1_ps(iteration));

        // for the newly escaped pixels, set the escape iteration
        // color is iteration if mask = 1, else the color value remains the same 
        colors_vec =_mm256_blendv_ps(colors_vec, _mm256_set1_ps(iteration), escaped_this_round_mask);

        // update pixels that escaped this iteration
        escaped_so_far_mask = _mm256_or_ps(escaped_so_far_mask, escaped_this_round_mask);

        // next fractal step
		fractal_avxf_julia(z_real, z_imag, z_real, z_imag, c_real, c_imag);
	}

    // check which pixels didn't escape
    __m256 max_iter_mask = _mm256_cmp_ps(_mm256_load_ps(color_array), _mm256_set1_ps(max_iterations), _CMP_GE_OQ);

    // give the pixels that didn't escape a color value of 0
    colors_vec =_mm256_blendv_ps(colors_vec, _mm256_set1_ps(0), max_iter_mask);

    // store the pixel values
    _mm256_store_ps(color_array, colors_vec);
}

void
fractal_avxf_get_colors(HCMATRIX hCmatrix, struct FractalProperties * fp)
{
}

/*
void
fractal_get_single_color(FRACDTYPE * color, FRACDTYPE x, FRACDTYPE y, FRACDTYPE _Complex (*fractal)(FRACDTYPE complex, FRACDTYPE _Complex), FRACDTYPE _Complex c, FRACDTYPE R, int max_iterations)
{
	int num_iterations = 0;
	FRACDTYPE _Complex z = x + y*I;

	for (; num_iterations < max_iterations; ++num_iterations) {
		if (fractal_escape_magnitude_check(z, R))
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

*/