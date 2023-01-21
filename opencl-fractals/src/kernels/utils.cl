bool
fractal_escape_magnitude_check(float z_real, float z_imag, float R)
{
    return (z_real*z_real + z_imag*z_imag) > (R*R);
}
