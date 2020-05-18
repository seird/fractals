#ifndef __BENCHMARK_H__
#define __BENCHMARK_H__


#include <stdio.h>
#include <time.h>


#define BENCH_FUNC(function) void function(void)

#define BENCH_RUN(function, iterations) do {\
    float time_start = (float) clock()/CLOCKS_PER_SEC;\
    for (int timeit_i=0; timeit_i < iterations; ++timeit_i) {\
        function();\
    }\
    float time_elapsed = (float) clock()/CLOCKS_PER_SEC - time_start;\
    printf("%s\n\t%10f seconds per run [%f seconds total]\n\n", #function, time_elapsed/iterations, time_elapsed);\
} while (0)



#endif
