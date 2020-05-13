#include "main.h"


#if (!defined(TEST) && !defined(SHARED))
int
main(int argc, char * argv[])
{    
    float c_real = -0.835f;
	float c_imag = -0.2321f;

    float _Complex c = -0.835 - 0.2321*I;
    float R = ceilf(cabs(c)) + 1;

    int ROWS = 1000;
    int COLS = 1000;

    HCMATRIX hCmatrix = fractal_cmatrix_create(1, 1);
    hCmatrix = fractal_cmatrix_reshape(hCmatrix, ROWS, COLS);

    float x_start = -R;
    float x_end = R;

    float y_start = -R;
    float y_end = R;

    float x_step = (x_end - x_start) / ROWS;
    float y_step = (y_end - y_start) / COLS;

    int max_iterations = 100;

    //fractal_get_colors_cmpx(hCmatrix, x_start, x_step, y_start, y_step, FRAC_JULIA, c, R, max_iterations);
    fractal_get_colors(hCmatrix, x_start, x_step, y_start, y_step, FRAC_JULIA, c_real, c_imag, R, max_iterations);
    //fractal_get_colors_th(hCmatrix, x_start, x_step, y_start, y_step, FRAC_JULIA, c_real, c_imag, R, max_iterations, 6);

    //float max_color = fractal_get_max_color(hCmatrix);

    fractal_cmatrix_free(hCmatrix);
	return 0;
}
#endif
