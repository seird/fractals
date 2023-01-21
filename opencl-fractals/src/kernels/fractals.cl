/**
 * z^2 + c
 */
void
fractal_z2(float * result_real, float * result_imag,
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
void
fractal_z3(float * result_real, float * result_imag,
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
void
fractal_z4(float * result_real, float * result_imag,
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
void
fractal_zconj2(float * result_real, float * result_imag,
               const float z_real, const float z_imag,
               const float c_real, const float c_imag)
{
    fractal_z2(result_real, result_imag,
                    z_real, -z_imag,
                    c_real, c_imag);
}


/**
 * (conj(z))^3 + c
 */
void
fractal_zconj3(float * result_real, float * result_imag,
               const float z_real, const float z_imag,
               const float c_real, const float c_imag)
{
    fractal_z3(result_real, result_imag,
                    z_real, -z_imag,
                    c_real, c_imag);
}


/**
 * (conj(z))^4 + c
 */
void
fractal_zconj4(float * result_real, float * result_imag,
               const float z_real, const float z_imag,
               const float c_real, const float c_imag)
{
    fractal_z4(result_real, result_imag,
                    z_real, -z_imag,
                    c_real, c_imag);
}


/**
 * (abs(z_real) + abs(c_real)*j)^2 + c
 */
void
fractal_zabs2(float * result_real, float * result_imag,
              const float z_real, const float z_imag,
              const float c_real, const float c_imag)
{
    fractal_z2(result_real, result_imag,
                    fabs(z_real), fabs(z_imag),
                    c_real, c_imag);
}


/**
 * (abs(z_real) + abs(c_real)*j)^3 + c
 */
void
fractal_zabs3(float * result_real, float * result_imag,
              const float z_real, const float z_imag,
              const float c_real, const float c_imag)
{
    fractal_z3(result_real, result_imag,
                    fabs(z_real), fabs(z_imag),
                    c_real, c_imag);
}


/**
 * (abs(z_real) + abs(c_real)*j)^4 + c
 */
void
fractal_zabs4(float * result_real, float * result_imag,
              const float z_real, const float z_imag,
              const float c_real, const float c_imag)
{
    fractal_z4(result_real, result_imag,
                    fabs(z_real), fabs(z_imag),
                    c_real, c_imag);
}


/**
 * [(z^2 + c - 1)/(2z + c - 2)]^2
 */
void
fractal_magnet(float * result_real, float * result_imag,
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
void
fractal_z2_z(float * result_real, float * result_imag,
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
