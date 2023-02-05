#include "benchmarks.h"


int MAX_ITERATIONS   = 2000;
int HEIGHT           = 2048; // multiple of vector size (16 for AVX512) and NUM_THREADS
int WIDTH            = 2048; // multiple of vector size (16 for AVX512) and NUM_THREADS
float C_REAL         = -0.788485;//-0.835f
float C_IMAG         = 0.004913;//-0.2321f
int NUM_THREADS      = 24;
enum FC_Mode MODE       = FC_MODE_JULIA;
enum FC_Fractal FRACTAL = FC_FRAC_Z2;

int
main(void)
{
    int num_runs = 10;

    printf("\n=================================================\nBenchmarking ...\n");
    printf("\tNumber of runs     = %20d\n", num_runs);
    printf("\tFractal iterations = %20d\n", MAX_ITERATIONS);
    printf("\tNumber of threads  = %20d\n", NUM_THREADS);
    printf("\tHEIGHT             = %20d\n", HEIGHT);
    printf("\tWIDTH              = %20d\n", WIDTH);
    printf("\tC_REAL             = %20f\n", C_REAL);
    printf("\tC_IMAG             = %20f\n", C_IMAG);
    printf("\tMODE               = %20d\n", MODE);
    printf("\tFRACTAL            = %20d\n", FRACTAL);
    putchar('\n');

    BENCH_RUN(bench_default, num_runs);
    BENCH_RUN(bench_threaded, num_runs);

    #ifdef __AVX2__
    BENCH_RUN(bench_avx, num_runs);
    BENCH_RUN(bench_avx_threaded, num_runs);
    #endif // __AVX2__

    #ifdef __AVX512DQ__
    BENCH_RUN(bench_avx512, num_runs);
    BENCH_RUN(bench_avx512_threaded, num_runs);
    #endif // __AVX512DQ__

    #ifdef CUDA
    fractal_cuda_init(WIDTH, HEIGHT);
    BENCH_RUN(bench_cuda, num_runs);
    BENCH_RUN(bench_cuda_lyapunov, num_runs);
    fractal_cuda_clean();
    #endif

    #ifdef OPENCL
    fractal_opencl_init(WIDTH, HEIGHT);
    BENCH_RUN(bench_opencl, num_runs);
    fractal_opencl_clean();
    #endif

    return 0;
}
