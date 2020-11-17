#include "main.h"


bool
fractal_escape_magnitude_check(float _Complex z, float R)
{
	return (crealf(z) * crealf(z) + cimagf(z) * cimagf(z)) > (R * R);
}

void
fractal_get_single_color(float * color, float x, float y, fractal_t fractal, float _Complex c, float R, int max_iterations)
{
	int num_iterations = 0;
	float _Complex z = x + y*I;

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

    float _Complex c = fp->c_real + fp->c_imag * I;
    fractal_t fractal = fractal_get(fp->frac);

    float y = fp->y_start;
    for (int row=0; row<hc->ROWS; ++row) {
        float x = fp->x_start;
        for (int col=0; col<hc->COLS; ++col) {
            fractal_get_single_color(&hc->cmatrix[row][col], x, y, fractal, c, fp->R, fp->max_iterations);
            x += fp->x_step;
        }
        y += fp->y_step;
    }
}
