__kernel void
lyapunov(__global uint * M, const int width, const int height, __global struct FractalProperties * fp, __global char * sequence, const int len)
{
    uint _x = get_global_id(0);
    uint _y = get_global_id(1);

    float a = fp->x_start + (float)_x/width*(fp->x_end - fp->x_start);
    float b = fp->y_start + (float)_y/height *(fp->y_end - fp->y_start);
   
    float lyapunov_exponent = 0.0f;
    float x = 0.5f;
    // lyapunov_exponent = (1/N) sum_n(log|r_n(1-2x_n|); n = 1..N
    for (int n=1; n<=fp->max_iterations; ++n) {
        float r = sequence[n%len] == 'A' ? a : b;
        x = r*x*(1-x);
        lyapunov_exponent += log(fabs(r*(1-2*x)));
    }

    // lyapunov_exponent /= fp->max_iterations;

    uchar value = lyapunov_exponent > 0.0f ? 0 : lyapunov_exponent * -1; // > 0 -> chaos, < 0 -> stable

    M[_y * width + _x]  = value;
}
