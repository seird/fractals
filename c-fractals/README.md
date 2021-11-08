# Compute fractal colors

## Build dynamic lirary

    $ make library

## Build static lirary

    $ make static

## Usage

Include the library interface **fracal_color.h** in the project. Add the source or link the dll.

### Functions

- HCMATRIX fractal_cmatrix_create(int height, int width);
- HCMATRIX fractal_cmatrix_reshape(HCMATRIX hCmatrix, int height_new, int width_new);
- void fractal_cmatrix_free(HCMATRIX hCmatrix);
- float * fractal_cmatrix_value(HCMATRIX hCmatrix, int height, int width);
- void fractal_get_colors(HCMATRIX hCmatrix, struct FractalProperties * fp);
- void fractal_get_colors_th(HCMATRIX hCmatrix, struct FractalProperties * fp, int num_threads);
- void fractal_avxf_get_colors(HCMATRIX hCmatrix, struct FractalProperties * fp);
- void fractal_avxf_get_colors_th(HCMATRIX hCmatrix, struct FractalProperties * fp, int num_threads);
- float fractal_cmatrix_max(HCMATRIX hCmatrix);
- void fractal_value_to_color(uint8_t * r, uint8_t * g, uint8_t * b, int value, enum FC_Color color);
- colorfunc_t fractal_colorfunc_get(enum FC_Color color);
- void fractal_cmatrix_save(HCMATRIX hCmatrix, const char * filename, enum FC_Color color);
- void fractal_image_save(uint8_t * image, int width, int height, const char * filename);
- uint8_t * fractal_image_create(int height, int width);
- void fractal_image_free(uint8_t * image);

## Examples

See `examples/`


```c
int
main(void)
{    
    /* ----------- INPUT PARAMETERS ----------- */
    float c_real = -0.7835f;
	float c_imag = -0.2321f;
    
    int width = 800;
    int height = 600;

    float R = 2;
    
    float aspect_ratio = (float)width/height;

    float x_start = -R;
    float x_end   =  R;

    float y_start = x_start/aspect_ratio;
    float y_end = x_end/aspect_ratio;    

    int max_iterations = 1000;

    enum FC_Mode mode = FC_MODE_JULIA;
    enum FC_Fractal fractal = FC_FRAC_Z2;
    enum FC_Color color = FC_COLOR_JET;
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
        .color = color,

        .c_real = c_real,
        .c_imag = c_imag,
        .R = R,
        .max_iterations = max_iterations,
    };

    HCMATRIX hCmatrix = fractal_cmatrix_create(height, width);

    fractal_avxf_get_colors(hCmatrix, &fp);
    fractal_cmatrix_save(hCmatrix, "fractal_image.png", fp.color);
    
    fractal_cmatrix_free(hCmatrix);

	return 0;
}
```

## Run and test

    $ make run
    $ make test
