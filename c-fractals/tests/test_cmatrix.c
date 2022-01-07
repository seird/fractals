#include "tests.h"


MU_TEST(test_cmatrix)
{   
    int height = 55;
    int width = 44;

    HCMATRIX hCmatrix = fractal_cmatrix_create(height, width);
    
    MU_CHECK(height == fractal_cmatrix_height(hCmatrix));
    MU_CHECK(width == fractal_cmatrix_width(hCmatrix));

    fractal_cmatrix_free(hCmatrix);
}
