#include "main.h"

#ifdef CUDA
#include "../../cuda-fractals/include/fractal_cuda.h"
#endif

#if (!defined(TEST) && !defined(SHARED) && !defined(STATIC) && !defined(BENCHMARK))
int
main(void)
{    
    /* ----------- INPUT PARAMETERS ----------- */
    float c_real = -0.7835f;
	float c_imag = -0.2321f;
    float _Complex c = c_real + c_imag*I;
    float R = ceilf(cabs(c)) + 1;
    
    int height = 10*108;
    int width = 8*240;
    
    float aspect_ratio = (float)width/height;

    float x_start = -R/5+0.5;
    float x_end   =  R/5+0.5;

    float y_start = x_start/aspect_ratio;
    float y_end = x_end/aspect_ratio;    

    int max_iterations = 1000;

    enum FC_Mode mode = FC_MODE_JULIA;
    enum FC_Fractal fractal = FC_FRAC_Z2;
    /* ---------------------------------------- */


    struct FractalProperties fp = {
        .x_start = x_start,
        .x_end = x_end,
        .y_start = y_start,
        .y_end = y_end,

        .width = width,
        .height = height,

        .frac = fractal,
        .mode = mode,
        .c_real = c_real,
        .c_imag = c_imag,
        .R = R,
        .max_iterations = max_iterations,
    };

    float counter = 0;
    int N = 100;
    char savename[20];

    for (int i=0; i<N; ++i) {
        c_real = 0.7885f * cosf(counter / (2 * 3.1416f));
        c_imag = 0.7885f * sinf(counter / (2 * 3.1416f));
        counter += 0.1f;
        printf("[%02d] c_real = %f, c_imag = %f\n", i, c_real, c_imag);

        fp.c_real = c_real;
        fp.c_imag = c_imag;
        sprintf(savename, "fractal_%02d.png", i);

    #ifdef CUDA
        int * image = fractal_image_create(height, width);
        fractal_cuda_init(width, height);
        
        fractal_cuda_get_colors(image, &fp);
        fractal_image_save(image, width, height, savename, FC_COLOR_ULTRA);

        fractal_cuda_clean();
        fractal_image_free(image);
    #else
        HCMATRIX hCmatrix = fractal_cmatrix_create(height, width);

        //fractal_get_colors(hCmatrix, &fp);
        //fractal_get_colors_th(hCmatrix, &fp, 10);
        //fractal_avxf_get_colors(hCmatrix, &fp);
        fractal_avxf_get_colors_th(hCmatrix, &fp, 10);

        //float max_color = fractal_cmatrix_max(hCmatrix);

        fractal_cmatrix_save(hCmatrix, savename, FC_COLOR_ULTRA);

        fractal_cmatrix_free(hCmatrix);
    #endif

    }

	return 0;
}
#endif
