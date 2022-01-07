#include "tests.h"


int tests_run = 0;
int tests_failed = 0;
int tests_result = 0;


static void
ALL_TESTS()
{
    MU_RUN_TEST(test_threaded_result);
    MU_RUN_TEST(test_cmatrix);
    

    #ifdef __AVX2__
    MU_RUN_TEST(test_avx_julia);
    MU_RUN_TEST(test_avx_julia_n);
    MU_RUN_TEST(test_avx_escape);
    MU_RUN_TEST(test_avx_vector_color);
    MU_RUN_TEST(test_avx_conj_n);
    MU_RUN_TEST(test_avx_abs_n);
    #endif // __AVX2__
}

int
main(void)
{
    ALL_TESTS();

    MU_STATS();

    return tests_failed != 0;
}