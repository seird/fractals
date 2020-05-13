#ifndef __MINUNIT_H__
#define __MINUNIT_H__


#include <stdio.h>


extern int tests_run;
extern int tests_failed;
extern int tests_result;


#define MU_TEST(function) void function(void)

#define MU_CHECK(test) do {\
    if (!(test)) {\
        printf("%s failed:\n\t%s:%d : %s\n\n", __func__, __FILE__, __LINE__, #test);\
        tests_result = 1;\
    }\
} while (0)

#define MU_ASSERT(message, test) do {\
    if (!(test)) {\
        printf("%s failed:\n\t%s:%d : %s\n\t%s\n\n", __func__, __FILE__, __LINE__, #test, message);\
        tests_result = 1;\
    }\
} while (0)

#define MU_RUN_TEST(test) do {\
    tests_result = 0;\
    test();\
    tests_failed += tests_result;\
    tests_run++;\
} while (0)

#define MU_STATS() printf("%d tests, %d failed\n\n", tests_run, tests_failed)

#endif