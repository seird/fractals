#include "buddha.h"
#include "main.h"

/*
 * Compute the trajectory in the complex plane
 *
 * Returns the length of the trajectory
 * The return value is 0 if it didn't escape
 * */
static int
get_trajectory(float _Complex* trajectory, fractal_t fractal, float _Complex c, float R, int max_iterations)
{
    float _Complex z = 0;
    for (int n = 0; n < max_iterations; ++n) {
        z = (*fractal)(z, c);

        // Add z to the trajectory
        trajectory[n] = z;

        if (fractal_escape_magnitude_check(z, R))
            return n;
    }

    return 0;
}

static void
update_visits(float** visits, int height, int width, float _Complex* trajectory, int size, float x_start, float x_end, float y_start, float y_end)
{
    // width x height corresponds to the resulting image size

    for (int i = 0; i < size; ++i) {
        float _Complex z = trajectory[i];

        // Map z in the complex plane to a integer index corresponding to a region of the image
        int w = (crealf(z) - x_start) / (x_end - x_start) * width;
        int h = (cimagf(z) - y_start) / (y_end - y_start) * height;

        if (w < 0 || w >= width || h < 0 || h >= height)
            continue;

        ++visits[h][w];
    }
}

void
buddha_get_colors(HS_CMATRIX hc, struct FractalProperties* fp)
{
    float _Complex trajectory[fp->max_iterations];
    float visits[hc->height][hc->width]; // Count the number of visits made by a z-trajectory in a region of the image
    fractal_t fractal = fractal_get(fp->frac);

    float x_step = (fp->x_end - fp->x_start) / fp->buddha.real_steps;
    float y_step = (fp->y_end - fp->y_start) / fp->buddha.imag_steps;

    for (float c_real = fp->x_start; c_real < fp->x_end; c_real += x_step) {
        for (float c_imag = fp->y_start; c_imag < fp->y_end; c_imag += y_step) {
            float _Complex c = c_real + c_imag * I;
            int n = get_trajectory(trajectory, fractal, c, fp->R, fp->max_iterations);
            if (!n)
                continue;

            // increment visits for the z-trajectory
            update_visits(hc->cmatrix, hc->height, hc->width, trajectory, n, fp->x_start, fp->x_end, fp->y_start, fp->y_end);
        }
        printf("\r%.1f%%", 100 * (c_real - fp->x_start) / (fp->x_end - fp->x_start));
        fflush(stdout);
    }

    float m = fractal_cmatrix_max(hc);
    for (int h = 0; h < hc->height; ++h) {
        for (int w = 0; w < hc->width; ++w) {
            hc->cmatrix[h][w] *= 255.f / m;
        }
    }
}
