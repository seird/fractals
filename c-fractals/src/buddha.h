#ifndef __BUDDHA_H__
#define __BUDDHA_H__

/* Note that to zoom into the complex plane in Buddha mode should still sample the comlex plane in the -2..2 (or large enough) range
 * To achieve this the real_steps / imag_steps should be inscreased?
 * */
#include "compute_avx.h"
#include "fractals_avx.h"
#include "main.h"

extern pthread_mutex_t mutex_buddha;

void
buddha_get_colors(HS_CMATRIX hc, struct FractalProperties* fp);

#ifdef __AVX2__

void
buddha_avxf_get_trajectory(HS_CMATRIX hc, fractal_avx_t fractal, __m256* c_real, __m256* c_imag, __m256* R, struct FractalProperties* fp);

void
buddha_avxf_update_visits(HS_CMATRIX hc, float* trajectory_real, float* trajectory_imag, float n_arr[VECFSIZE], struct FractalProperties* fp);

#endif // __AVX2__

#endif // __BUDDHA_H__
