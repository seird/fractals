#include "main.h"


#if (!defined(TEST) && !defined(SHARED))
int
main(int argc, char * argv[])
{    
    double c_real = -0.835f;
	double c_imag = -0.2321f;

    double _Complex c = -0.835 - 0.2321*I;
    double R = ceilf(cabs(c)) + 1;

    int ROWS = 1000;
    int COLS = 1000;

    HCMATRIX hCmatrix = fractal_cmatrix_create(1, 1);
    hCmatrix = fractal_cmatrix_reshape(hCmatrix, ROWS, COLS);

    double x_start = -R;
    double x_end = R;

    double y_start = -R;
    double y_end = R;

    double x_step = (x_end - x_start) / ROWS;
    double y_step = (y_end - y_start) / COLS;

    int max_iterations = 100;
    
    struct FractalProperties fp = {
        .x_start = -R,
        .x_step = x_step,
        .y_start = y_start,
        .y_step = y_step,
        .frac = FRAC_JULIA,
        .c_real = c_real,
        .c_imag = c_imag,
        .R = R,
        .max_iterations = max_iterations,
    };

    fractal_get_colors(hCmatrix, &fp);
    //fractal_get_colors_th(hCmatrix, fp, 6);

    double max_color = fractal_get_max_color(hCmatrix);

    fractal_cmatrix_free(hCmatrix);
	return 0;
}
#endif
