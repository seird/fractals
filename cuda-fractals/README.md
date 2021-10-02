# Compute fractal colors with a CUDA device

## Build dynamic lirary

    $ make lib


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
    void fractal_cuda_get_colors(uint8_t * image, struct FractalProperties * fp);
    ```
