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

#ifdef __AVX2__

#include "compute_avx.h"

void
buddha_avxf_get_trajectory(HS_CMATRIX hc, fractal_avx_t fractal, __m256* c_real, __m256* c_imag, __m256* R, struct FractalProperties* fp)
{
    __m256 z_real = _mm256_set1_ps(0.0f);
    __m256 z_imag = _mm256_set1_ps(0.0f);

    float n_arr[VECFSIZE] __attribute__((aligned(AVX_ALIGNMENT)));
    float trajectory_real[fp->max_iterations * VECFSIZE] __attribute__((aligned(AVX_ALIGNMENT)));
    float trajectory_imag[fp->max_iterations * VECFSIZE] __attribute__((aligned(AVX_ALIGNMENT)));

    __m256 n = _mm256_set1_ps(0); // Stores the iteration at which the escaped occured -- this is the length of the trajectory
    __m256 escaped_so_far_mask = _mm256_set1_ps(0);
    __m256 escaped_mask;

    for (int i = 0; i < fp->max_iterations; ++i) {
        fractal(&z_real, &z_imag, &z_real, &z_imag, c_real, c_imag);

        // get all pixels that escaped this iteration
        fractal_avxf_escape_magnitude_check(&escaped_mask, &z_real, &z_imag, R);

        // get pixels that escaped for the first time
        __m256 escaped_this_iteration_mask = _mm256_and_ps(
          escaped_mask,
          _mm256_xor_ps(escaped_mask, escaped_so_far_mask));

        // for the newly escaped pixels, set the escape iteration
        // color is iteration if mask = 1, else the color value remains the same
        n = _mm256_blendv_ps(n, _mm256_set1_ps(i), escaped_this_iteration_mask);

        // update pixels that escaped this iteration
        escaped_so_far_mask = _mm256_or_ps(escaped_so_far_mask, escaped_this_iteration_mask);

        // store the z's in the trajectory arrays
        _mm256_store_ps(trajectory_real + i * VECFSIZE, z_real);
        _mm256_store_ps(trajectory_imag + i * VECFSIZE, z_imag);

        // abort if all pixels have escaped
        if (_mm256_movemask_ps(escaped_so_far_mask) == 255)
            return;
    }

    // update the visits
    _mm256_store_ps(n_arr, n);
    buddha_avxf_update_visits(hc, trajectory_real, trajectory_imag, n_arr, fp);
}

void
buddha_avxf_update_visits(HS_CMATRIX hc, float* trajectory_real, float* trajectory_imag, float n_arr[VECFSIZE], struct FractalProperties* fp)
{
    float x_factor = hc->width / (fp->x_end - fp->x_start);
    float y_factor = hc->height / (fp->y_end - fp->y_start);

    for (int i = 0; i < VECFSIZE; ++i) {
        if (!(int)n_arr[i])
            continue;

        for (int n = 0; n < fp->max_iterations; ++n) {
            int w = (trajectory_real[n * VECFSIZE + i] - fp->x_start) * x_factor;
            int h = (trajectory_imag[n * VECFSIZE + i] - fp->y_start) * y_factor;

            if (w < 0 || w >= hc->width || h < 0 || h >= hc->height)
                continue;

            ++hc->cmatrix[h][w];
        }
    }
}

#endif // __AVX2__
