#include "tests.h"


#ifdef __AVX2__

MU_TEST(test_avx_conj_n)
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

    __m256 z_real_n_vec = _mm256_load_ps(z_real_array);
    __m256 z_imag_n_vec = _mm256_load_ps(z_imag_array);
    __m256 c_real_n_vec = _mm256_load_ps(c_real_array);
    __m256 c_imag_n_vec = _mm256_load_ps(c_imag_array);

    // Compute the reference julia result
    __m256 result_real_vec, result_imag_vec;
    fractal_avxf_zconj2(&result_real_vec, &result_imag_vec, &z_real_vec, &z_imag_vec, &c_real_vec, &c_imag_vec);

    // Compute the julia_n result for n = 2
    __m256 result_real_n_vec, result_imag_n_vec;
    fractal_avxf_zconjn(&result_real_n_vec, &result_imag_n_vec, &z_real_n_vec, &z_imag_n_vec, &c_real_n_vec, &c_imag_n_vec, 2);

    for (int i=0; i<VECFSIZE; ++i) {
        // Compare the reference julia result with the julia_n result
        MU_CHECK_FLT_EQ(((float *)&result_real_n_vec)[i], ((float *)&result_real_vec)[i]);
        MU_CHECK_FLT_EQ(((float *)&result_imag_n_vec)[i], ((float *)&result_imag_vec)[i]);
    }
}

#endif // __AVX2__
