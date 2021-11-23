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

    fp->_x_step = (fp->x_end - fp->x_start) / fp->width;
    fp->_y_step = (fp->y_end - fp->y_start) / fp->height;

    float _Complex c = fp->c_real + fp->c_imag * I;
    fractal_t fractal = fractal_get(fp->frac);

    switch (fp->mode)
    {
        case FC_MODE_JULIA:
        {
            float y = fp->y_start;
            for (int h=0; h<hc->height; ++h) {
                float x = fp->x_start;
                for (int w=0; w<hc->width; ++w) {
                    fractal_get_single_color(&hc->cmatrix[h][w], x, y, fractal, c, fp->R, fp->max_iterations);
                    x += fp->_x_step;
                }
                y += fp->_y_step;
            }
            break;
        }
        case FC_MODE_FLAMES:
        {
            srand(time(NULL));

            flames_get_colors(fp); // all variations are currently used
            break;
        }
        case FC_MODE_MANDELBROT:
        case FC_MODE_LYAPUNOV:
        default: printf("Unsupported mode.\n");
    }
}
