#pragma once

#include "julia.cuh"
#include "mandelbrot.cuh"
#include "lyapunov.cuh"


#ifdef __linux__
extern "C" bool fractal_cuda_init(int width, int height);
extern "C" void fractal_cuda_clean();
extern "C" void fractal_cuda_get_colors(uint8_t * image, struct FractalProperties * fp);
#else
extern "C" bool __declspec(dllexport) fractal_cuda_init(int width, int height);
extern "C" void __declspec(dllexport) fractal_cuda_clean();
extern "C" void __declspec(dllexport) fractal_cuda_get_colors(uint8_t * image, struct FractalProperties * fp);
#endif
