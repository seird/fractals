#ifndef __BUDDHA_H__
#define __BUDDHA_H__

#include "main.h"

void
buddha_get_colors(HS_CMATRIX hc, struct FractalProperties* fp);

#ifdef __AVX2__

void
buddha_avxf_get_trajectory(HS_CMATRIX hc, fractal_avx_t fractal, __m256* c_real, __m256* c_imag, __m256* R, struct FractalProperties* fp);

void
buddha_avxf_update_visits(HS_CMATRIX hc, float* trajectory_real, float* trajectory_imag, float n_arr[VECFSIZE], struct FractalProperties* fp);

#endif // __AVX2__

#endif // __BUDDHA_H__
