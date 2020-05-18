[![pipeline status](https://gitlab.com/kdries/opengl-fractals/badges/opt-intrinsics/pipeline.svg)](https://gitlab.com/kdries/opengl-fractals/commits/opt-intrinsics)


# Compute fractal colors

## Build dynamic lirary

    $ make library

## Usage

Include the library interface **fracal_color.h** in the project. Add the source or link the dll.

### Functions


- Initialize a color matrix, returns a handle **HCMATRIX**

    ```c
    HCMATRIX fractal_cmatrix_create(int ROWS, int COLS);
    ```

- Reshape an existing color matrix

    ```c
    HCMATRIX fractal_cmatrix_reshape(HCMATRIX hCmatrix, int ROWS_new, int COLS_new);
    ```

- Free a color matrix

    ```c
    void fractal_cmatrix_free(HCMATRIX hCmatrix);
    ```

- Retrieve a pointer to a value in the color matrix

    ```c
    FRACDTYPE * fractal_cmatrix_value(HCMATRIX hCmatrix, int row, int col);
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
    FRACDTYPE fractal_cmatrix_max(HCMATRIX hCmatrix);
    ```

### Structs

- Parameters to pass to `fractal_get_colors`

    ```c
    struct FractalProperties {
        FRACDTYPE x_start;
        FRACDTYPE x_step;
        FRACDTYPE y_start;
        FRACDTYPE y_step;
        enum Fractal frac;
        FRACDTYPE c_real;
        FRACDTYPE c_imag;
        FRACDTYPE R;
        int max_iterations;
    };
    ```

### Fractals

```c
enum Fractal {
    FRAC_JULIA,
};
```

## Run and test

    $ make run
    $ make test
