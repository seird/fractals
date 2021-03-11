#include "fractal.cuh"


int * d_image;


extern "C" bool
fractal_cuda_init(int width, int height)
{
    if (d_image != NULL) {
        return false;
    } else {
        return (cudaMalloc(&d_image, sizeof(int) * width*height) == cudaSuccess);
    }
}

extern "C" void
fractal_cuda_clean()
{
    cudaFree(&d_image);
    d_image = NULL;
}
