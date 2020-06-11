#include "main.h"


#if (!defined(TEST) && !defined(SHARED) && !defined(BENCHMARK))
int
main(int argc, char * argv[])
{    
    FRACDTYPE c_real = -0.7835f;
	FRACDTYPE c_imag = -0.2321f;

    FRACDTYPE _Complex c = c_real + c_imag*I;
    FRACDTYPE R = ceilf(cabs(c)) + 1;

    int ROWS = 16*100;
    int COLS = 16*100;

    HCMATRIX hCmatrix = fractal_cmatrix_create(1, 1);
    hCmatrix = fractal_cmatrix_reshape(hCmatrix, ROWS, COLS);

    FRACDTYPE x_start = -R;
    FRACDTYPE x_end = R;

    FRACDTYPE y_start = -R;
    FRACDTYPE y_end = R;

    FRACDTYPE x_step = (x_end - x_start) / ROWS;
    FRACDTYPE y_step = (y_end - y_start) / COLS;

    int max_iterations = 1000;
    
    struct FractalProperties fp = {
        .x_start = -R,
        .x_step = x_step,
        .y_start = y_start,
        .y_step = y_step,
        .frac = FRAC_Z2,
        .mode = MODE_JULIA,
        .c_real = c_real,
        .c_imag = c_imag,
        .R = R,
        .max_iterations = max_iterations,
    };

    //fractal_get_colors(hCmatrix, &fp);
    //fractal_get_colors_th(hCmatrix, &fp, 6);
    fractal_avxf_get_colors(hCmatrix, &fp);
    //fractal_avxf_get_colors_th(hCmatrix, &fp, 6);

    //FRACDTYPE max_color = fractal_cmatrix_max(hCmatrix);

    fractal_cmatrix_save(hCmatrix, "fractal.png");

    fractal_cmatrix_free(hCmatrix);

	return 0;
}
#endif
