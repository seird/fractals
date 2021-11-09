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

- Create an image array.

    ```c
    uint8_t * fractal_image_create(int height, int width);
    ```

- Free an image array.

    ```c
    void fractal_image_free(uint8_t * image);
    ```

- Do the color computation

    ```c
    void fractal_cuda_get_colors(uint8_t * image, struct FractalProperties * fp);
    ```

- Save an image as png

    ```c
    void fractal_image_save(uint8_t * image, int width, int height, const char * filename);
    ```
