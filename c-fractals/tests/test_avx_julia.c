#include "tests.h"


#ifdef __AVX2__

MU_TEST(test_avx_julia)
{   
    // Test vectors
    float z_real_array[VECFSIZE] = { -0.835, -1.5 , 1.835 , 1.5   , 0.1414, 2.1414, 0    , -3.1   };
    float z_imag_array[VECFSIZE] = { -3.1  , 0    , 2.1414, 0.1414, 1.5   , 1.835 , -1.5 , -0.835 };

    float c_real_array[VECFSIZE] = { -0.835, 0    , -1.1 , -2.3   , 1.4151, 5.444 , 0.835, 1.835 };
    float c_imag_array[VECFSIZE] = { 1.835 , 0.835, 5.444, 1.4151 , -2.3  , -1.1  , 0    , -0.835};

    // Load the avx vectors
    __m256 z_real_vec = _mm256_load_ps(z_real_array);
    __m256 z_imag_vec = _mm256_load_ps(z_imag_array);
    __m256 c_real_vec = _mm256_load_ps(c_real_array);
    __m256 c_imag_vec = _mm256_load_ps(c_imag_array);

    // Compute the avx result
    __m256 result_real_vec, result_imag_vec;
    fractal_avxf_z2(&result_real_vec, &result_imag_vec, &z_real_vec, &z_imag_vec, &c_real_vec, &c_imag_vec);

    for (int i=0; i<VECFSIZE; ++i) {
        // Compute the reference result
        float _Complex z = z_real_array[i] + z_imag_array[i] * I;
        float _Complex c = c_real_array[i] + c_imag_array[i] * I;
        float _Complex result = fractal_z2(z, c);

        // Compare the reference result with the avx result
        MU_CHECK_FLT_EQ_ERROR(crealf(result), ((float *)&result_real_vec)[i], 0.00001);
        MU_CHECK_FLT_EQ_ERROR(cimagf(result), ((float *)&result_imag_vec)[i], 0.00001);
    }
}

#endif // __AVX2__
