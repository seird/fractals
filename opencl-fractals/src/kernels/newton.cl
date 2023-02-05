__kernel void
newton(__global uint * M, const int width, const int height, __global struct FractalProperties * fp)
{
    uint x = get_global_id(0);
    uint y = get_global_id(1);

    float z_real = fp->x_start + (float)x/width * (fp->x_end - fp->x_start);
    float z_imag = fp->y_start + (float)y/height * (fp->y_end - fp->y_start);

    float r_real;
    float r_imag;

    float a_real = fp->c_real;
    float a_imag = fp->c_imag;

    int i = 0;

    float bailout = 0.000001f;

    for (; i<fp->max_iterations; ++i) {
        switch (fp->frac) {
            case FC_FRAC_N_Z3: {
                fractal_newton_z3(&r_real, &r_imag, z_real, z_imag, a_real, a_imag);

                // this could all be done better
                int num_roots = 3;
                float roots[6] = {
                    1.0f, 0.0f, 
                    -0.5f, 0.5f * sqrt(3.0f), 
                    -0.5f, -0.5f * sqrt(3.0f)
                };

                for (int r=0; r<num_roots; ++r) {
                    float root_real = roots[r*2];
                    float root_imag = roots[r*2+1];
                    if (fabs(r_real - root_real) < bailout && fabs(r_imag - root_imag) < bailout) {
                        M[y * (width) + x] = i == fp->max_iterations ? 0 : i;
                        return;
                    }
                }

                break;
            }
            case FC_FRAC_N_SIN: {
                fractal_newton_sin(&r_real, &r_imag, z_real, z_imag, a_real, a_imag);

                // reduce current value to the 0..2*pi range
                float root_real = fmod(r_real, 2*M_PI_F);
                float root_imag = fmod(r_imag, 2*M_PI_F);

                if (fabs(root_real) < bailout && fabs(root_imag) < bailout) {
                    M[y * (width) + x] = i == fp->max_iterations ? 0 : i;
                    return;
                }
                
                break;
            }
            case FC_FRAC_N_SIN1: {
                bailout = 0.001f;
                fractal_newton_sin1(&r_real, &r_imag, z_real, z_imag, a_real, a_imag);

                // reduce current value to the 0..2*pi range
                float root_real = fmod(r_real, 2*M_PI_F);
                float root_imag = fmod(r_imag, 2*M_PI_F);
                
                if (fabs(root_real - M_PI_2) < bailout && fabs(root_imag) < bailout) {
                    M[y * (width) + x] = i == fp->max_iterations ? 0 : i;
                    return;
                }

                break;
            }
            default:
                break;
        }

        z_real = r_real;
        z_imag = r_imag;
    }

    M[y * (width) + x]  = i == fp->max_iterations ? 0 : i;
}
