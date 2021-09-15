#include "kernels.cuh"


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
