#include "tests.h"


MU_TEST(test_avx_vector_color)
{
    float __attribute__((aligned(AVX_ALIGNMENT))) colors_avx[VECFSIZE] = {0};
    float colors[VECFSIZE] = {0};

    float R = 2.0;
    int max_iterations = 10;

    // Test vectors
    float x_array[VECFSIZE] = { -2.0, -1.5, -1.111, -0.0102, 0.1414, 1.4414, 1.5, 2.0 };
    float y_array[VECFSIZE] = { -2.0, -1.5, -1.111, -0.0102, 0.1414, 1.4414, 1.5, 2.0 };
    float c_real_array[VECFSIZE] = { -0.835, 0    , -1.1 , -2.3   , 1.4151, 5.444 , 0.835, 1.835 };
    float c_imag_array[VECFSIZE] = { 1.835 , 0.835, 5.444, 1.4151 , -2.3  , -1.1  , 0    , -0.835};

    // Load the avx vectors
    __m256 c_real_vec = _mm256_load_ps(c_real_array);
    __m256 c_imag_vec = _mm256_load_ps(c_imag_array);
    __m256 x_vec = _mm256_load_ps(x_array);
    __m256 y_vec = _mm256_load_ps(y_array);

    __m256 RR = _mm256_set1_ps(R*R);

    void (*fractal_avx)(__m256 *, __m256 *, __m256 *, __m256 *, __m256 * , __m256 *) = fractal_avx_get(FRAC_Z2);

    fractal_avxf_get_vector_color(colors_avx, &x_vec, &y_vec, &c_real_vec, &c_imag_vec, &RR, max_iterations, fractal_avx);
    
    float _Complex (*fractal)(float complex, float _Complex) = fractal_get(FRAC_Z2);

    for (int i=0; i<VECFSIZE; ++i) {
        // Compute the reference result
        float _Complex c = c_real_array[i] + c_imag_array[i] * I;
        fractal_get_single_color(&colors[i], x_array[i], y_array[i], fractal, c, R, max_iterations);

        // Compare the reference result with the avx result
        MU_CHECK(colors[i] == colors_avx[i]);
    }
}