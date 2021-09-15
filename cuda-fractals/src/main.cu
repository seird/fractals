#include "main.cuh"


uint8_t * d_image;


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
    if (d_image != NULL) {
        cudaFree(&d_image);
        d_image = NULL;
    }
}


fractal_cuda_kernel_t kernels_julia[FC_FRAC_NUM_ENTRIES] = {
    fractal_cuda_kernel_julia_z2,
    fractal_cuda_kernel_julia_z3,
    fractal_cuda_kernel_julia_z4,
    fractal_cuda_kernel_julia_zconj2,
    fractal_cuda_kernel_julia_zconj3,
    fractal_cuda_kernel_julia_zconj4,
    fractal_cuda_kernel_julia_zabs2,
    fractal_cuda_kernel_julia_zabs3,
    fractal_cuda_kernel_julia_zabs4,
    fractal_cuda_kernel_julia_magnet,
    fractal_cuda_kernel_julia_z2_z,
};

fractal_cuda_kernel_t kernels_mandelbrot[FC_FRAC_NUM_ENTRIES] = {
    fractal_cuda_kernel_mandelbrot_z2,
    fractal_cuda_kernel_mandelbrot_z3,
    fractal_cuda_kernel_mandelbrot_z4,
    fractal_cuda_kernel_mandelbrot_zconj2,
    fractal_cuda_kernel_mandelbrot_zconj3,
    fractal_cuda_kernel_mandelbrot_zconj4,
    fractal_cuda_kernel_mandelbrot_zabs2,
    fractal_cuda_kernel_mandelbrot_zabs3,
    fractal_cuda_kernel_mandelbrot_zabs4,
    fractal_cuda_kernel_mandelbrot_magnet,
    fractal_cuda_kernel_mandelbrot_z2_z,
};


extern "C" void
fractal_cuda_get_colors(uint8_t * image, struct FractalProperties * fp)
{
    dim3 threads(8, 8);
    dim3 blocks(fp->width / threads.x, fp->height / threads.y);

    // Get the kernel function
    fractal_cuda_kernel_t kernel;
    switch (fp->mode) {
        case FC_MODE_JULIA:
            kernel = kernels_julia[fp->frac];
            break;
        case FC_MODE_MANDELBROT:
            kernel = kernels_mandelbrot[fp->frac];
            break;
        case FC_MODE_LYAPUNOV:
            char * d_sequence;
            cudaMalloc(&d_sequence, sizeof(char) * fp->sequence_length);
            cudaMemcpy(d_sequence, fp->sequence, sizeof(char) * fp->sequence_length, cudaMemcpyHostToDevice);

            fractal_cuda_kernel_lyapunov<<<blocks, threads>>>(d_image,
                                                              fp->x_start, fp->x_end,
                                                              fp->y_start, fp->y_end,
                                                              fp->width, fp->height,
                                                              d_sequence, fp->sequence_length,
                                                              fp->max_iterations,
                                                              fp->color);
            cudaFree(&d_sequence);
            goto finish;
        default:
            kernel = kernels_julia[fp->frac];
    }

    // Do the cuda computation
    kernel<<<blocks, threads>>>(d_image,
                                fp->x_start, fp->x_end,
                                fp->y_start, fp->y_end,
                                fp->c_real, fp->c_imag,
                                fp->width, fp->height,
                                fp->max_iterations, fp->R,
                                fp->color);

    finish:
    cudaThreadSynchronize();

    // Copy the result back
    cudaMemcpy(image, d_image, sizeof(int) * fp->width*fp->height, cudaMemcpyDeviceToHost);
}
