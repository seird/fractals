#include "fractal.cuh"
#include "gradients.cuh"


/** multiple kernel functions are used for better performance
 * - lookup table to function in a kernel is x8 slower
 * - ? pass device pointers
 */


// powf, cpowf etc are much slower

/**
 * z^2 + c
 */
__device__ void
fractal_cuda_z2(float * result_real, float * result_imag,
                const float z_real, const float z_imag,
                const float c_real, const float c_imag)
{
    // z^2 = (a+bj)*(a+bj) = a*a - b*b + 2(a*b)j
    *result_real = z_real*z_real - z_imag*z_imag;
    *result_imag = 2*z_real*z_imag;

    // z^2 + c
    *result_real += c_real;
    *result_imag += c_imag;
}


/**
 * z^3 + c
 */
__device__ void
fractal_cuda_z3(float * result_real, float * result_imag,
                const float z_real, const float z_imag,
                const float c_real, const float c_imag)
{
    // z^4 = (a+bj)*(a+bj)*(a+bj)
    //     = (a*a - b*b + 2(a*b)j) * (a+bj)
    //     = a^3 - 3*a*b^2 + (3*a^2*b-b^3)*j
    //     = a^2*a - 3*a*b^2 + (3*a^2*b-b^2*b)j
    float z_real_2 = z_real*z_real;
    float z_imag_2 = z_imag*z_imag;

    *result_real = z_real_2*z_real - 3*z_real*z_imag_2;
    *result_imag = 3*z_real_2*z_imag-z_imag_2*z_imag;

    // z^3 + c
    *result_real += c_real;
    *result_imag += c_imag;
}


/**
 * z^4 + c
 */
__device__ void
fractal_cuda_z4(float * result_real, float * result_imag,
                const float z_real, const float z_imag,
                const float c_real, const float c_imag)
{
    // z^4 = (a+bj)*(a+bj)*(a+bj)*(a+bj)
    //     = a^4 + b^4 - 6 * a^2*b^2 + 4*(a^3*b - a*b^3)*j
    //     = a^2*a^2 + b^2*b^2 - 6 * a^2*b^2 + 4*(a^2*a*b - a*b*b^2)j
    float z_real_2 = z_real*z_real;
    float z_imag_2 = z_imag*z_imag;

    *result_real = z_real_2*z_real_2 + z_imag_2*z_imag_2 - 6*z_real_2*z_imag_2;
    *result_imag = 4*(z_real_2*z_real*z_imag - z_real*z_imag*z_imag_2);

    // z^4 + c
    *result_real += c_real;
    *result_imag += c_imag;
}


/**
 * (conj(z))^2 + c
 */
__device__ void
fractal_cuda_zconj2(float * result_real, float * result_imag,
                    const float z_real, const float z_imag,
                    const float c_real, const float c_imag)
{
    fractal_cuda_z2(result_real, result_imag,
                    z_real, -z_imag,
                    c_real, c_imag);
}


/**
 * (conj(z))^3 + c
 */
__device__ void
fractal_cuda_zconj3(float * result_real, float * result_imag,
                    const float z_real, const float z_imag,
                    const float c_real, const float c_imag)
{
    fractal_cuda_z3(result_real, result_imag,
                    z_real, -z_imag,
                    c_real, c_imag);
}


/**
 * (conj(z))^4 + c
 */
__device__ void
fractal_cuda_zconj4(float * result_real, float * result_imag,
                    const float z_real, const float z_imag,
                    const float c_real, const float c_imag)
{
    fractal_cuda_z4(result_real, result_imag,
                    z_real, -z_imag,
                    c_real, c_imag);
}


/**
 * (abs(z_real) + abs(c_real)*j)^2 + c
 */
__device__ void
fractal_cuda_zabs2(float * result_real, float * result_imag,
                   const float z_real, const float z_imag,
                   const float c_real, const float c_imag)
{
    fractal_cuda_z2(result_real, result_imag,
                    fabsf(z_real), fabsf(z_imag),
                    c_real, c_imag);
}


/**
 * (abs(z_real) + abs(c_real)*j)^3 + c
 */
__device__ void
fractal_cuda_zabs3(float * result_real, float * result_imag,
                   const float z_real, const float z_imag,
                   const float c_real, const float c_imag)
{
    fractal_cuda_z3(result_real, result_imag,
                    fabsf(z_real), fabsf(z_imag),
                    c_real, c_imag);
}


/**
 * (abs(z_real) + abs(c_real)*j)^4 + c
 */
__device__ void
fractal_cuda_zabs4(float * result_real, float * result_imag,
                   const float z_real, const float z_imag,
                   const float c_real, const float c_imag)
{
    fractal_cuda_z4(result_real, result_imag,
                    fabsf(z_real), fabsf(z_imag),
                    c_real, c_imag);
}


/**
 * [(z^2 + c - 1)/(2z + c - 2)]^2
 */
__device__ void
fractal_cuda_magnet(float * result_real, float * result_imag,
                    const float z_real, const float z_imag,
                    const float c_real, const float c_imag)
{
    // numerator
    // z^2 = (a+bj)*(a+bj) = a^2 - b^2 + 2abj
    float num_real = z_real*z_real - z_imag*z_imag + c_real - 1;
    float num_imag = 2*z_real*z_imag + c_imag;

    // denominator
    float denom_real = 2*z_real + c_real - 2;
    float denom_imag = 2*z_imag + c_imag;

    // num/denom = (a+bj)/(c+dj) = [(a+bj)*(c-dj)]/[(c+dj)*(c-dj)]
    //           = (a*c - a*d*j + b*c*j + b*d)/(c^2+d^2)
    //           = (a*c + b*d + (b*c-a*d)j)/(c^2+d^2)
    float denom_sqsum = denom_real*denom_real + denom_imag*denom_imag;
    float frac_real = (num_real*denom_real + num_imag*denom_imag)/denom_sqsum;
    float frac_imag = (num_imag*denom_real - num_real*denom_imag)/denom_sqsum;

    // []^2
    *result_real = frac_real*frac_real - frac_imag*frac_imag;
    *result_imag = 2*frac_real*frac_imag;
}


/**
 *z^2 + c/z
 */
__device__ void
fractal_cuda_z2_z(float * result_real, float * result_imag,
                  const float z_real, const float z_imag,
                  const float c_real, const float c_imag)
{
    // z^2 = (a+bj)*(a+bj) = a*a - b*b + 2(a*b)j
    // c/z = (a+bj)/(c+dj) = [(a+bj)*(c-dj)]/[(c+dj)*(c-dj)]
    //                     = (a*c - a*d*j + b*c*j + b*d)/(c^2+d^2)
    //                     = (a*c + b*d + (b*c-a*d)j)/(c^2+d^2)

    float z_real_2 = z_real*z_real;
    float z_imag_2 = z_imag*z_imag;
    float denom = z_real_2 + z_imag_2;

    *result_real = z_real_2 - z_imag_2 + (c_real*z_real + c_imag*z_imag) / denom;
    *result_imag = 2*z_real*z_imag + (c_imag*z_real - c_real*z_imag) / denom;
}


__device__ bool
fractal_cuda_escape_magnitude_check(float z_real, float z_imag, float R)
{
    return (z_real*z_real + z_imag*z_imag) > (R*R);
}

__device__ void
fractal_cuda_kernel_julia(uint8_t * colors, const float w_start, const float w_end,
                                    const float h_start, const float h_end,
                                    const float c_real, const float c_imag,
                                    int width, int height,
                                    int max_iterations, float R,
                                    fractal_cuda_t fractal,
                                    FC_Color color)
{
    int w = (blockIdx.x*blockDim.x) + threadIdx.x;
    int h = (blockIdx.y*blockDim.y) + threadIdx.y;

    float z_real = w_start + (float)w/width * (w_end - w_start);
    float z_imag = h_start + (float)h/height * (h_end - h_start);

    int num_iterations = 0;
    float r_real;
    float r_imag;
    for (; num_iterations<max_iterations; ++num_iterations) {
        if (fractal_cuda_escape_magnitude_check(z_real, z_imag, R)) {
            break;
        }
        fractal(&r_real, &r_imag, z_real, z_imag, c_real, c_imag);
        z_real = r_real;
        z_imag = r_imag;
    }

    int value = num_iterations == max_iterations ? BLACK : num_iterations;

    colorfuncs[color%FC_COLOR_NUM_ENTRIES](
        colors + h*width*3 + w*3,
        colors + h*width*3 + w*3 + 1,
        colors + h*width*3 + w*3 + 2,
        value
    );
}


__global__ void
fractal_cuda_kernel_julia_z2(uint8_t * colors, const float w_start, const float w_end,
                                       const float h_start, const float h_end,
                                       const float c_real, const float c_imag,
                                       int width, int height,
                                       int max_iterations, float R,
                                       FC_Color color)
{
    fractal_cuda_kernel_julia(colors, w_start, w_end,
                                        h_start, h_end,
                                        c_real, c_imag,
                                        width, height,
                                        max_iterations, R,
                                        fractal_cuda_z2,
                                        color);
}

__global__ void
fractal_cuda_kernel_julia_z3(uint8_t * colors, const float w_start, const float w_end,
                                       const float h_start, const float h_end,
                                       const float c_real, const float c_imag,
                                       int width, int height,
                                       int max_iterations, float R,
                                       FC_Color color)
{
    fractal_cuda_kernel_julia(colors, w_start, w_end,
                                        h_start, h_end,
                                        c_real, c_imag,
                                        width, height,
                                        max_iterations, R,
                                        fractal_cuda_z3,
                                        color);
}

__global__ void
fractal_cuda_kernel_julia_z4(uint8_t * colors, const float w_start, const float w_end,
                                       const float h_start, const float h_end,
                                       const float c_real, const float c_imag,
                                       int width, int height,
                                       int max_iterations, float R,
                                       FC_Color color)
{
    fractal_cuda_kernel_julia(colors, w_start, w_end,
                                        h_start, h_end,
                                        c_real, c_imag,
                                        width, height,
                                        max_iterations, R,
                                        fractal_cuda_z4,
                                        color);
}

__global__ void
fractal_cuda_kernel_julia_zconj2(uint8_t * colors, const float w_start, const float w_end,
                                           const float h_start, const float h_end,
                                           const float c_real, const float c_imag,
                                           int width, int height,
                                           int max_iterations, float R,
                                           FC_Color color)
{
    fractal_cuda_kernel_julia(colors, w_start, w_end,
                                        h_start, h_end,
                                        c_real, c_imag,
                                        width, height,
                                        max_iterations, R,
                                        fractal_cuda_zconj2,
                                        color);
}

__global__ void
fractal_cuda_kernel_julia_zconj3(uint8_t * colors, const float w_start, const float w_end,
                                           const float h_start, const float h_end,
                                           const float c_real, const float c_imag,
                                           int width, int height,
                                           int max_iterations, float R,
                                           FC_Color color)
{
    fractal_cuda_kernel_julia(colors, w_start, w_end,
                                        h_start, h_end,
                                        c_real, c_imag,
                                        width, height,
                                        max_iterations, R,
                                        fractal_cuda_zconj3,
                                        color);
}

__global__ void
fractal_cuda_kernel_julia_zconj4(uint8_t * colors, const float w_start, const float w_end,
                                           const float h_start, const float h_end,
                                           const float c_real, const float c_imag,
                                           int width, int height,
                                           int max_iterations, float R,
                                           FC_Color color)
{
    fractal_cuda_kernel_julia(colors, w_start, w_end,
                                        h_start, h_end,
                                        c_real, c_imag,
                                        width, height,
                                        max_iterations, R,
                                        fractal_cuda_zconj4,
                                        color);
}

__global__ void
fractal_cuda_kernel_julia_zabs2(uint8_t * colors, const float w_start, const float w_end,
                                          const float h_start, const float h_end,
                                          const float c_real, const float c_imag,
                                          int width, int height,
                                          int max_iterations, float R,
                                          FC_Color color)
{
    fractal_cuda_kernel_julia(colors, w_start, w_end,
                                        h_start, h_end,
                                        c_real, c_imag,
                                        width, height,
                                        max_iterations, R,
                                        fractal_cuda_zabs2,
                                        color);
}

__global__ void
fractal_cuda_kernel_julia_zabs3(uint8_t * colors, const float w_start, const float w_end,
                                          const float h_start, const float h_end,
                                          const float c_real, const float c_imag,
                                          int width, int height,
                                          int max_iterations, float R,
                                          FC_Color color)
{
    fractal_cuda_kernel_julia(colors, w_start, w_end,
                                        h_start, h_end,
                                        c_real, c_imag,
                                        width, height,
                                        max_iterations, R,
                                        fractal_cuda_zabs3,
                                        color);
}

__global__ void
fractal_cuda_kernel_julia_zabs4(uint8_t * colors, const float w_start, const float w_end,
                                          const float h_start, const float h_end,
                                          const float c_real, const float c_imag,
                                          int width, int height,
                                          int max_iterations, float R,
                                          FC_Color color)
{
    fractal_cuda_kernel_julia(colors, w_start, w_end,
                                        h_start, h_end,
                                        c_real, c_imag,
                                        width, height,
                                        max_iterations, R,
                                        fractal_cuda_zabs4,
                                        color);
}

__global__ void
fractal_cuda_kernel_julia_magnet(uint8_t * colors, const float w_start, const float w_end,
                                           const float h_start, const float h_end,
                                           const float c_real, const float c_imag,
                                           int width, int height,
                                           int max_iterations, float R,
                                           FC_Color color)
{
    fractal_cuda_kernel_julia(colors, w_start, w_end,
                                        h_start, h_end,
                                        c_real, c_imag,
                                        width, height,
                                        max_iterations, R,
                                        fractal_cuda_magnet,
                                        color);
}

__global__ void
fractal_cuda_kernel_julia_z2_z(uint8_t * colors, const float w_start, const float w_end,
                                         const float h_start, const float h_end,
                                         const float c_real, const float c_imag,
                                         int width, int height,
                                         int max_iterations, float R,
                                         FC_Color color)
{
    fractal_cuda_kernel_julia(colors, w_start, w_end,
                                        h_start, h_end,
                                        c_real, c_imag,
                                        width, height,
                                        max_iterations, R,
                                        fractal_cuda_z2_z,
                                        color);
}


// Mandelbrot

__device__ void
fractal_cuda_kernel_mandelbrot(uint8_t * colors, const float w_start, const float w_end,
                                const float h_start, const float h_end,
                                float c_real, float c_imag,
                                int width, int height,
                                int max_iterations, float R,
                                fractal_cuda_t fractal,
                                FC_Color color)
{
    int w = (blockIdx.x*blockDim.x) + threadIdx.x;
    int h = (blockIdx.y*blockDim.y) + threadIdx.y;

    c_real = w_start + (float)w/width * (w_end - w_start);
    c_imag = h_start + (float)h/height * (h_end - h_start);
    float z_real = 0;
    float z_imag = 0;

    int num_iterations = 0;
    float r_real;
    float r_imag;
    for (; num_iterations<max_iterations; ++num_iterations) {
        if (fractal_cuda_escape_magnitude_check(z_real, z_imag, R)) {
            break;
        }
        fractal(&r_real, &r_imag, z_real, z_imag, c_real, c_imag);
        z_real = r_real;
        z_imag = r_imag;
    }

    int value = num_iterations == max_iterations ? BLACK : num_iterations;

    colorfuncs[color%FC_COLOR_NUM_ENTRIES](
        colors + h*width*3 + w*3,
        colors + h*width*3 + w*3 + 1,
        colors + h*width*3 + w*3 + 2,
        value
    );
}

__global__ void
fractal_cuda_kernel_mandelbrot_z2(uint8_t * colors, const float w_start, const float w_end,
                                       const float h_start, const float h_end,
                                       const float c_real, const float c_imag,
                                       int width, int height,
                                       int max_iterations, float R,
                                       FC_Color color)
{
    fractal_cuda_kernel_mandelbrot(colors, w_start, w_end,
                                        h_start, h_end,
                                        c_real, c_imag,
                                        width, height,
                                        max_iterations, R,
                                        fractal_cuda_z2,
                                        color);
}

__global__ void
fractal_cuda_kernel_mandelbrot_z3(uint8_t * colors, const float w_start, const float w_end,
                                       const float h_start, const float h_end,
                                       const float c_real, const float c_imag,
                                       int width, int height,
                                       int max_iterations, float R,
                                       FC_Color color)
{
    fractal_cuda_kernel_mandelbrot(colors, w_start, w_end,
                                        h_start, h_end,
                                        c_real, c_imag,
                                        width, height,
                                        max_iterations, R,
                                        fractal_cuda_z3,
                                        color);
}

__global__ void
fractal_cuda_kernel_mandelbrot_z4(uint8_t * colors, const float w_start, const float w_end,
                                       const float h_start, const float h_end,
                                       const float c_real, const float c_imag,
                                       int width, int height,
                                       int max_iterations, float R,
                                       FC_Color color)
{
    fractal_cuda_kernel_mandelbrot(colors, w_start, w_end,
                                        h_start, h_end,
                                        c_real, c_imag,
                                        width, height,
                                        max_iterations, R,
                                        fractal_cuda_z4,
                                        color);
}

__global__ void
fractal_cuda_kernel_mandelbrot_zconj2(uint8_t * colors, const float w_start, const float w_end,
                                           const float h_start, const float h_end,
                                           const float c_real, const float c_imag,
                                           int width, int height,
                                           int max_iterations, float R,
                                           FC_Color color)
{
    fractal_cuda_kernel_mandelbrot(colors, w_start, w_end,
                                        h_start, h_end,
                                        c_real, c_imag,
                                        width, height,
                                        max_iterations, R,
                                        fractal_cuda_zconj2,
                                        color);
}

__global__ void
fractal_cuda_kernel_mandelbrot_zconj3(uint8_t * colors, const float w_start, const float w_end,
                                           const float h_start, const float h_end,
                                           const float c_real, const float c_imag,
                                           int width, int height,
                                           int max_iterations, float R,
                                           FC_Color color)
{
    fractal_cuda_kernel_mandelbrot(colors, w_start, w_end,
                                        h_start, h_end,
                                        c_real, c_imag,
                                        width, height,
                                        max_iterations, R,
                                        fractal_cuda_zconj3,
                                        color);
}

__global__ void
fractal_cuda_kernel_mandelbrot_zconj4(uint8_t * colors, const float w_start, const float w_end,
                                           const float h_start, const float h_end,
                                           const float c_real, const float c_imag,
                                           int width, int height,
                                           int max_iterations, float R,
                                           FC_Color color)
{
    fractal_cuda_kernel_mandelbrot(colors, w_start, w_end,
                                        h_start, h_end,
                                        c_real, c_imag,
                                        width, height,
                                        max_iterations, R,
                                        fractal_cuda_zconj4,
                                        color);
}

__global__ void
fractal_cuda_kernel_mandelbrot_zabs2(uint8_t * colors, const float w_start, const float w_end,
                                          const float h_start, const float h_end,
                                          const float c_real, const float c_imag,
                                          int width, int height,
                                          int max_iterations, float R,
                                          FC_Color color)
{
    fractal_cuda_kernel_mandelbrot(colors, w_start, w_end,
                                        h_start, h_end,
                                        c_real, c_imag,
                                        width, height,
                                        max_iterations, R,
                                        fractal_cuda_zabs2,
                                        color);
}

__global__ void
fractal_cuda_kernel_mandelbrot_zabs3(uint8_t * colors, const float w_start, const float w_end,
                                          const float h_start, const float h_end,
                                          const float c_real, const float c_imag,
                                          int width, int height,
                                          int max_iterations, float R,
                                          FC_Color color)
{
    fractal_cuda_kernel_mandelbrot(colors, w_start, w_end,
                                        h_start, h_end,
                                        c_real, c_imag,
                                        width, height,
                                        max_iterations, R,
                                        fractal_cuda_zabs3,
                                        color);
}

__global__ void
fractal_cuda_kernel_mandelbrot_zabs4(uint8_t * colors, const float w_start, const float w_end,
                                          const float h_start, const float h_end,
                                          const float c_real, const float c_imag,
                                          int width, int height,
                                          int max_iterations, float R,
                                          FC_Color color)
{
    fractal_cuda_kernel_mandelbrot(colors, w_start, w_end,
                                        h_start, h_end,
                                        c_real, c_imag,
                                        width, height,
                                        max_iterations, R,
                                        fractal_cuda_zabs4,
                                        color);
}

__global__ void
fractal_cuda_kernel_mandelbrot_magnet(uint8_t * colors, const float w_start, const float w_end,
                                           const float h_start, const float h_end,
                                           const float c_real, const float c_imag,
                                           int width, int height,
                                           int max_iterations, float R,
                                           FC_Color color)
{
    fractal_cuda_kernel_mandelbrot(colors, w_start, w_end,
                                        h_start, h_end,
                                        c_real, c_imag,
                                        width, height,
                                        max_iterations, R,
                                        fractal_cuda_magnet,
                                        color);
}

__global__ void
fractal_cuda_kernel_mandelbrot_z2_z(uint8_t * colors, const float w_start, const float w_end,
                                         const float h_start, const float h_end,
                                         const float c_real, const float c_imag,
                                         int width, int height,
                                         int max_iterations, float R,
                                         FC_Color color)
{
    fractal_cuda_kernel_mandelbrot(colors, w_start, w_end,
                                        h_start, h_end,
                                        c_real, c_imag,
                                        width, height,
                                        max_iterations, R,
                                        fractal_cuda_z2_z,
                                        color);
}


// Lyapunov

__global__ void
fractal_cuda_kernel_lyapunov(uint8_t * colors, const float w_start, const float w_end,
                             const float h_start, const float h_end,
                             int width, int height,
                             char * sequence, size_t sequence_length,
                             int max_iterations,
                             FC_Color color)
{
    int w = (blockIdx.x*blockDim.x) + threadIdx.x;
    int h = (blockIdx.y*blockDim.y) + threadIdx.y;

    float a = w_start + (float)w/width * (w_end - w_start);
    float b = h_start + (float)h/height * (h_end - h_start);
   
    float lyapunov_exponent = 0.0f;
    float x_n = 0.5;
    // lyapunov_exponent = (1/N) sum_n(log|r_n(1-2x_n|); n = 1..N
    for (int n=1; n<=max_iterations; ++n) {
        float r_n = sequence[n%sequence_length] == 'A' ? a : b;
        x_n = r_n*x_n*(1-x_n);
        lyapunov_exponent += logf(fabsf(r_n*(1-2*x_n)));
    }

    // lyapunov_exponent /= max_iterations;

    int value = lyapunov_exponent > 0.0f ? 0 : (int)lyapunov_exponent * -1; // > 0 -> chaos, < 0 -> stable
    
    colorfuncs[color%FC_COLOR_NUM_ENTRIES](
        colors + h*width*3 + w*3,
        colors + h*width*3 + w*3 + 1,
        colors + h*width*3 + w*3 + 2,
        value
    );
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
