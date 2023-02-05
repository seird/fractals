__kernel void
mandelbrot(__global uint * M, const int width, const int height, __global struct FractalProperties * fp)
{
    uint x = get_global_id(0);
    uint y = get_global_id(1);

    fp->c_real = fp->x_start + (float)x/width * (fp->x_end - fp->x_start);
    fp->c_imag = fp->y_start + (float)y/height * (fp->y_end - fp->y_start);
    float z_real = 0;
    float z_imag = 0;

    int num_iterations = 0;
    float r_real;
    float r_imag;

    int i = 0;
    for (; i<fp->max_iterations; ++i) {
        if (fractal_escape_magnitude_check(z_real, z_imag, fp->R)) {
            break;
        }

        switch (fp->frac) {
            case FC_FRAC_Z2:
                fractal_z2(&r_real, &r_imag, z_real, z_imag, fp->c_real, fp->c_imag);
                break;
            case FC_FRAC_Z3:
                fractal_z3(&r_real, &r_imag, z_real, z_imag, fp->c_real, fp->c_imag);
                break;
            case FC_FRAC_Z4:
                fractal_z4(&r_real, &r_imag, z_real, z_imag, fp->c_real, fp->c_imag);
                break;
            case FC_FRAC_ZCONJ2:
                fractal_zconj2(&r_real, &r_imag, z_real, z_imag, fp->c_real, fp->c_imag);
                break;
            case FC_FRAC_ZCONJ3:
                fractal_zconj3(&r_real, &r_imag, z_real, z_imag, fp->c_real, fp->c_imag);
                break;
            case FC_FRAC_ZCONJ4:
                fractal_zconj4(&r_real, &r_imag, z_real, z_imag, fp->c_real, fp->c_imag);
                break;
            case FC_FRAC_ZABS2:
                fractal_zabs2(&r_real, &r_imag, z_real, z_imag, fp->c_real, fp->c_imag);
                break;
            case FC_FRAC_ZABS3:
                fractal_zabs3(&r_real, &r_imag, z_real, z_imag, fp->c_real, fp->c_imag);
                break;
            case FC_FRAC_ZABS4:
                fractal_zabs4(&r_real, &r_imag, z_real, z_imag, fp->c_real, fp->c_imag);
                break;
            case FC_FRAC_MAGNET:
                fractal_magnet(&r_real, &r_imag, z_real, z_imag, fp->c_real, fp->c_imag);
                break;
            case FC_FRAC_Z2_Z:
                fractal_z2_z(&r_real, &r_imag, z_real, z_imag, fp->c_real, fp->c_imag);
                break;
            default:
                break;
        }
        z_real = r_real;
        z_imag = r_imag;
    }

    M[y * width + x] = i == fp->max_iterations ? 0 : i;
}
