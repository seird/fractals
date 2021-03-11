# Compute fractal colors with a CUDA device

## Build dynamic lirary

    $ make

## Usage

Include the library interface **fracal_cuda.h** and **fracal_color.h** in the project. Add the source or link the dll and define CUDA.

### Functions


- Allocate the required memory on the CUDA device

    ```c
    bool fractal_cuda_init(int width, int height);
    ```

- Free the allocated memory

    ```c
    void fractal_cuda_clean();
    ```

- Do the color computation

    ```c
    void fractal_cuda_get_colors(int * image, struct FractalProperties * fp);
    ```
