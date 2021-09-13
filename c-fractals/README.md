# Compute fractal colors

## Build dynamic lirary

    $ make library

## Build static lirary

    $ make static

## Usage

Include the library interface **fracal_color.h** in the project. Add the source or link the dll.

### Functions


- Initialize a color matrix, returns a handle **HCMATRIX**

    ```c
    HCMATRIX fractal_cmatrix_create(int height, int width);
    ```

- Reshape an existing color matrix

    ```c
    HCMATRIX fractal_cmatrix_reshape(HCMATRIX hCmatrix, int height_new, int width_new);
    ```

- Free a color matrix

    ```c
    void fractal_cmatrix_free(HCMATRIX hCmatrix);
    ```

- Retrieve a pointer to a value in the color matrix

    ```c
    float * fractal_cmatrix_value(HCMATRIX hCmatrix, int height, int width);
    ```

- Compute the fractal colors

    ```c
    void fractal_get_colors(HCMATRIX hCmatrix, struct FractalProperties * fp);
    ```

- Compute the fractal colors with threads

    ```c
    void fractal_get_colors_th(HCMATRIX hCmatrix, struct FractalProperties * fp, int num_threads);
    ```

- Compute the fractal colors with AVX2

    ```c
    void fractal_avxf_get_colors(HCMATRIX hCmatrix, struct FractalProperties * fp);
    ```

- Compute the fractal colors with AVX2 and threads

    ```c
    void fractal_avxf_get_colors_th(HCMATRIX hCmatrix, struct FractalProperties * fp, int num_threads);
    ```

- Retrieve the maximum color value in the color matrix

    ```c
    float fractal_cmatrix_max(HCMATRIX hCmatrix);
    ```

- Convert a cmatrix value to rgb

    ```c
    void fractal_value_to_color(float * r, float * g, float * b, int value, enum FC_Color color);
    ```

- Save a color matrix as png

    ```c
    void fractal_cmatrix_save(HCMATRIX hCmatrix, const char * filename);
    ```

- Save an image array as png

    ```c
    void fractal_image_save(int * image, int width, int height, const char * filename, enum FC_Color color);
    ```

- Create an image array

    ```c
    int * fractal_image_create(int height, int width);
    ```

- Free an image array
    
    ```c
    void fractal_image_free(int * image);
    ```
    
### Structs

- Parameters to pass to `fractal_get_colors`

    ```c
    struct FractalProperties {
        float x_start;
        float x_end;
        float y_start;
        float y_end;
        float width;
        float height;
        enum FC_Fractal frac;
        enum FC_Mode mode;
        float c_real;
        float c_imag;
        float R;
        int max_iterations;
        char * lyapunov_sequence;
        size_t sequence_length;
    };
    ```

### Fractals

```c
enum FC_Fractal {
    FC_FRAC_Z2,     // z^2 + c
    FC_FRAC_Z3,     // z^3 + c
    FC_FRAC_Z4,     // z^4 + c
    FC_FRAC_ZCONJ2, // (conj(z))^2 + c
    FC_FRAC_ZCONJ3, // (conj(z))^3 + c
    FC_FRAC_ZCONJ4, // (conj(z))^4 + c
    FC_FRAC_ZABS2, // (abs(z_real) + abs(c_real)*j)^2 + c
    FC_FRAC_ZABS3, // (abs(z_real) + abs(c_real)*j)^3 + c
    FC_FRAC_ZABS4, // (abs(z_real) + abs(c_real)*j)^4 + c
};
```

### Modes

```c
enum FC_Mode {
    FC_MODE_MANDELBROT,
    FC_MODE_JULIA,
    FC_MODE_LYAPUNOV,
};
```

### Colors

```c
enum FC_Color {
    FC_COLOR_ULTRA,
    FC_COLOR_MONOCHROME,
    FC_COLOR_TRI,
    FC_COLOR_JET,
    FC_COLOR_LAVENDER,
    FC_COLOR_BINARY,
};
```

## Run and test

    $ make run
    $ make test
