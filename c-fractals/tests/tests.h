#ifndef __TESTS_H__
#define __TESTS_H__


#include "minunit.h"

#include <complex.h>
#include <math.h>

#include "../src/fractal_color.h"
#include "../src/compute_avx.h"


MU_TEST(test_threaded_result);
MU_TEST(test_avx_julia);

#endif