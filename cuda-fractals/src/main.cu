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


extern "C" void
fractal_cuda_get_colors(uint8_t * image, struct FractalProperties * fp)
{
    dim3 threads(8, 8);
    dim3 blocks(fp->width / threads.x, fp->height / threads.y);

    switch (fp->mode) {
        case FC_MODE_JULIA:
            julia_kernels[fp->frac % FC_FRAC_NUM_ENTRIES]<<<blocks, threads>>>(d_image, *fp);
            break;
        case FC_MODE_MANDELBROT:
            mandelbrot_kernels[fp->frac % FC_FRAC_NUM_ENTRIES]<<<blocks, threads>>>(d_image, *fp);
            break;
        case FC_MODE_LYAPUNOV:
            char * d_sequence;
            cudaMalloc(&d_sequence, sizeof(char) * fp->sequence_length);
            cudaMemcpy(d_sequence, fp->sequence, sizeof(char) * fp->sequence_length, cudaMemcpyHostToDevice);

            fractal_cuda_kernel_lyapunov<<<blocks, threads>>>(d_image, *fp, d_sequence);
            cudaFree(&d_sequence);
        default:
            break;
    }

    cudaThreadSynchronize();

    // Copy the result back
    cudaMemcpy(image, d_image, sizeof(int) * fp->width*fp->height, cudaMemcpyDeviceToHost);
}
