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
    double * fractal_cmatrix_value(HCMATRIX hCmatrix, int row, int col);
    ```

- Compute the fractal colors

    ```c
    void fractal_get_colors(HCMATRIX hCmatrix,
                           double x_start, double x_step, double y_start, double y_step,
                           enum Fractal frac, double c_real, double c_imag,
                           double R, int max_iterations);
    ```

- Compute the fractal colors with threads

    ```c
    void fractal_get_colors_th(HCMATRIX hCmatrix,
                            double x_start, double x_step, double y_start, double y_step,
                            enum Fractal frac, double c_real, double c_imag,
                            double R, int max_iterations, int num_threads);
    ```

- Retrieve the maximum color value in the color matrix

    ```c
    double fractal_get_max_color(HCMATRIX hCmatrix);
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
