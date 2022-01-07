#include "main.cuh"


uint8_t * d_image;
int d_width;
int d_height;


extern "C" bool
fractal_cuda_init(int width, int height)
{
    if (d_image != NULL) {
        return false;
    } else {
        d_width = width;
        d_height = height;
        return (cudaMalloc(&d_image, sizeof(uint8_t) * width*height*3) == cudaSuccess);
    }
}

extern "C" void
fractal_cuda_clean()
{
    if (d_image != NULL) {
        cudaFree(d_image);
        d_image = NULL;
        d_width = 0;
        d_height = 0;
    }
}


extern "C" void
fractal_cuda_get_colors(uint8_t * image, struct FractalProperties * fp)
{
    if (d_image == NULL || d_width == 0 || d_height == 0) return;
    
    dim3 threads(8, 8);
    dim3 blocks(d_width / threads.x, d_height / threads.y);

    switch (fp->mode) {
        case FC_MODE_JULIA:
            julia_kernels[fp->frac % FC_FRAC_NUM_ENTRIES]<<<blocks, threads>>>(d_image, d_width, d_height, *fp);
            break;
        case FC_MODE_MANDELBROT:
            mandelbrot_kernels[fp->frac % FC_FRAC_NUM_ENTRIES]<<<blocks, threads>>>(d_image, d_width, d_height, *fp);
            break;
        case FC_MODE_LYAPUNOV:
            char * d_sequence;
            cudaMalloc(&d_sequence, sizeof(char) * fp->sequence_length);
            cudaMemcpy(d_sequence, fp->sequence, sizeof(char) * fp->sequence_length, cudaMemcpyHostToDevice);

            fractal_cuda_kernel_lyapunov<<<blocks, threads>>>(d_image, d_width, d_height, *fp, d_sequence);
            cudaFree(&d_sequence);
        default:
            break;
    }

    // Copy the result back
    cudaMemcpy(image, d_image, sizeof(uint8_t) * d_width*d_height*3, cudaMemcpyDeviceToHost);
}
