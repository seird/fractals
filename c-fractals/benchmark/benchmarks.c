#include "benchmarks.h"


int MAX_ITERATIONS   = 1000;
int ROWS             = 1000;
int COLS             = 1000;
float C_REAL         = -0.788485;//-0.835f
float C_IMAG         = 0.004913;//-0.2321f
int NUM_THREADS      = 6;
enum Mode MODE       = MODE_JULIA;
enum Fractal FRACTAL = FRAC_Z2;

int
main(void)
{
    int num_runs = 10;

    printf("\n=================================================\nBenchmarking ...\n");
    printf("\tNumber of runs     = %20d\n", num_runs);
    printf("\tFractal iterations = %20d\n", MAX_ITERATIONS);
    printf("\tNumber of threads  = %20d\n", NUM_THREADS);
    printf("\tROWS               = %20d\n", ROWS);
    printf("\tCOLUMNS            = %20d\n", COLS);
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

    return 0;
}
