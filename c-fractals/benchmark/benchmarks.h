#ifndef __BENCHMARKS_H__
#define __BENCHMARKS_H__


#include "benchmark.h"
#include "../src/main.h"
#include "../include/fractal_color.h"

#ifdef CUDA
#include "../../cuda-fractals/include/fractal_cuda.h"
#endif

#ifdef OPENCL
#include "../../opencl-fractals/include/fractal_opencl.h"
#endif


extern int MAX_ITERATIONS;
extern int HEIGHT;
extern int WIDTH;
extern float C_REAL;
extern float C_IMAG;
extern int NUM_THREADS;
extern enum FC_Mode MODE;
extern enum FC_Fractal FRACTAL;


BENCH_FUNC(bench_default);
BENCH_FUNC(bench_threaded);
BENCH_FUNC(bench_avx);
BENCH_FUNC(bench_avx_threaded);
BENCH_FUNC(bench_avx512);
BENCH_FUNC(bench_avx512_threaded);
BENCH_FUNC(bench_cuda);
BENCH_FUNC(bench_cuda_lyapunov);
BENCH_FUNC(bench_opencl);


#endif
