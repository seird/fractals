#include "tests.h"


#ifdef __AVX2__

MU_TEST(test_avx_escape)
{
    // Test vectors
    float z_real_array[VECFSIZE] = { -0.835, -1.5 , 1.835 , 1.5   , 0.1414, 2.1414, 0    , -3.1   };
    float z_imag_array[VECFSIZE] = { -3.1  , 0    , 2.1414, 0.1414, 1.5   , 1.835 , -1.5 , -0.835 };

    __m256 z_real_vec = _mm256_load_ps(z_real_array);
    __m256 z_imag_vec = _mm256_load_ps(z_imag_array);

    __m256 escaped_mask;

    float R = 2.0;
    __m256 RR = _mm256_set1_ps(R*R);
    fractal_avxf_escape_magnitude_check(&escaped_mask, &z_real_vec, &z_imag_vec, &RR);
    
    int escaped_avx = _mm256_movemask_ps(escaped_mask); // 0bXXXXXXXX -- LSB => escaped_mask[0]

    for (int i=0; i<VECFSIZE; ++i) {
        // Compute the reference result
        float _Complex z = z_real_array[i] + z_imag_array[i] * I;
        bool escaped = fractal_escape_magnitude_check(z, R);

        // Compare the reference result with the avx result
        MU_CHECK(escaped == (escaped_avx & 1));
        escaped_avx >>= 1;
    }
}

#endif // __AVX2__
