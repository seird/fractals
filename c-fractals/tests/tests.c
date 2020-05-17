#include "tests.h"


int tests_run = 0;
int tests_failed = 0;
int tests_result = 0;


static void
ALL_TESTS()
{
    MU_RUN_TEST(test_threaded_result);
    MU_RUN_TEST(test_avx_julia);
}

int
main(int argc, char ** argv)
{
    ALL_TESTS();

    MU_STATS();

    return tests_failed != 0;
}