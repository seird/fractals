#include "../include/fractal_color.h"

int
main(void)
{
    /* ----------- INPUT PARAMETERS ----------- */
    float R = 2.0f;

    int height = 2048;
    int width = 2048;

    float x_start = -R;
    float x_end = R;

    float y_start = -R;
    float y_end = R;

    int max_iterations = 1000;

    enum FC_Mode mode = FC_MODE_BUDDHA;
    enum FC_Fractal fractal = FC_FRAC_Z2;
    /* enum FC_Color color = FC_COLOR_RAW; */
    enum FC_Color color = FC_COLOR_ULTRA;
    /* ---------------------------------------- */

    struct FractalProperties fp = {
        .x_start = x_start,
        .x_end = x_end,
        .y_start = y_start,
        .y_end = y_end,

        .frac = fractal,
        .mode = mode,
        .color = color,

        .R = R,
        .max_iterations = max_iterations,

        .buddha = {
          .real_steps = 10000,
          .imag_steps = 10000,
        },
    };

    HCMATRIX hCmatrix = fractal_cmatrix_create(height, width);

    fractal_avxf_get_colors(hCmatrix, &fp);

    fractal_cmatrix_save(hCmatrix, "buddhabrot_avx.png", color);

    fractal_cmatrix_free(hCmatrix);

    return 0;
}
