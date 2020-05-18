#ifndef __TESTS_H__
#define __TESTS_H__


#include "minunit.h"

#include <complex.h>
#include <math.h>
#include <stdbool.h>

#include "../src/fractal_color.h"
#include "../src/compute_avx.h"
#include "../src/fractals.h"
#include "../src/main.h"


MU_TEST(test_threaded_result);

MU_TEST(test_avx_julia);
MU_TEST(test_avx_escape);
MU_TEST(test_avx_vector_color);

#endif