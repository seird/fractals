#include "kernels.cuh"

__device__ void
fractal_cuda_kernel_mandelbrot(uint8_t * colors, FractalProperties fp, fractal_cuda_t fractal)
{
    int w = (blockIdx.x*blockDim.x) + threadIdx.x;
    int h = (blockIdx.y*blockDim.y) + threadIdx.y;

    fp.c_real = fp.x_start + (float)w/fp.width * (fp.x_end - fp.x_start);
    fp.c_imag = fp.y_start + (float)h/fp.height * (fp.y_end - fp.y_start);
    float z_real = 0;
    float z_imag = 0;

    int num_iterations = 0;
    float r_real;
    float r_imag;
    
    for (; num_iterations<fp.max_iterations; ++num_iterations) {
        if (fractal_cuda_escape_magnitude_check(z_real, z_imag, fp.R)) {
            break;
        }
        fractal(&r_real, &r_imag, z_real, z_imag, fp.c_real, fp.c_imag);
        z_real = r_real;
        z_imag = r_imag;
    }

    int value = num_iterations == fp.max_iterations ? BLACK : num_iterations;

    colorfuncs[fp.color%FC_COLOR_NUM_ENTRIES](
        colors + h*fp.width*3 + w*3,
        colors + h*fp.width*3 + w*3 + 1,
        colors + h*fp.width*3 + w*3 + 2,
        value
    );
}

__global__ void
fractal_cuda_kernel_mandelbrot_z2(uint8_t * colors, FractalProperties fp)
{
    fractal_cuda_kernel_mandelbrot(colors, fp, fractal_cuda_z2);
}

__global__ void
fractal_cuda_kernel_mandelbrot_z3(uint8_t * colors, FractalProperties fp)
{
    fractal_cuda_kernel_mandelbrot(colors, fp, fractal_cuda_z3);
}

__global__ void
fractal_cuda_kernel_mandelbrot_z4(uint8_t * colors, FractalProperties fp)
{
    fractal_cuda_kernel_mandelbrot(colors, fp, fractal_cuda_z4);
}

__global__ void
fractal_cuda_kernel_mandelbrot_zconj2(uint8_t * colors, FractalProperties fp)
{
    fractal_cuda_kernel_mandelbrot(colors, fp, fractal_cuda_zconj2);
}

__global__ void
fractal_cuda_kernel_mandelbrot_zconj3(uint8_t * colors, FractalProperties fp)
{
    fractal_cuda_kernel_mandelbrot(colors, fp, fractal_cuda_zconj3);
}

__global__ void
fractal_cuda_kernel_mandelbrot_zconj4(uint8_t * colors, FractalProperties fp)
{
    fractal_cuda_kernel_mandelbrot(colors, fp, fractal_cuda_zconj4);
}

__global__ void
fractal_cuda_kernel_mandelbrot_zabs2(uint8_t * colors, FractalProperties fp)
{
    fractal_cuda_kernel_mandelbrot(colors, fp, fractal_cuda_zabs2);
}

__global__ void
fractal_cuda_kernel_mandelbrot_zabs3(uint8_t * colors, FractalProperties fp)
{
    fractal_cuda_kernel_mandelbrot(colors, fp, fractal_cuda_zabs3);
}

__global__ void
fractal_cuda_kernel_mandelbrot_zabs4(uint8_t * colors, FractalProperties fp)
{
    fractal_cuda_kernel_mandelbrot(colors, fp, fractal_cuda_zabs4);
}

__global__ void
fractal_cuda_kernel_mandelbrot_magnet(uint8_t * colors, FractalProperties fp)
{
    fractal_cuda_kernel_mandelbrot(colors, fp, fractal_cuda_magnet);
}

__global__ void
fractal_cuda_kernel_mandelbrot_z2_z(uint8_t * colors, FractalProperties fp)
{
    fractal_cuda_kernel_mandelbrot(colors, fp, fractal_cuda_z2_z);
}


fractal_cuda_kernel_t mandelbrot_kernels[FC_FRAC_NUM_ENTRIES] = {
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
