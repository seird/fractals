#ifndef __BENCHMARKS_H__
#define __BENCHMARKS_H__


#include "benchmark.h"
#include "../src/main.h"
#include "../src/fractal_color.h"


extern int MAX_ITERATIONS;
extern int ROWS;
extern int COLS;
extern float C_REAL;
extern float C_IMAG;
extern int NUM_THREADS;
extern enum Mode MODE;
extern enum Fractal FRACTAL;


BENCH_FUNC(bench_default);
BENCH_FUNC(bench_threaded);
BENCH_FUNC(bench_avx);
BENCH_FUNC(bench_avx_threaded);


#endif
