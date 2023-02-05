// z_{n+1} = z_n - a * (f(z_n) / f'(z_n)) + c

// z*q = a*c - b*d +  (a*d + b*c)j
// z*z = a*a - b*b +  (a*b + a*b)j = a^2-b^2 + 2abj
// z/q = [(a+bj) * (c-dj)] / [(c+dj) * (c-dj)]
//     = (ac + bd)/(c^2 + d^2) + [(bc - ad)/c^2+d^2]j


/**
 * f(z) = z^3 - 1
 * f'(z) = 3z^2
 */
void
fractal_newton_z3(float * rr, float * ri,
                  const float zr, const float zi,
                  const float ar, const float ai)
{
    // ---- f(z) ----
    float fr = zr*zr*zr - 3*zr*zi*zi - 1;
    float fi = 3*zr*zr*zi - zi*zi*zi;

    // ---- f'(z) ----
    float dfr = 3 * (zr*zr - zi*zi);
    float dfi = 6*zr*zi;
    
    // ---- f(z) / f'(z) ----
    float denom = dfr*dfr + dfi*dfi;
    float fracr = (fr*dfr + fi*dfi) / denom;
    float fraci = (fi*dfr - fr*dfi) / denom;

    // ---- a * (f(z) / f'(z)) ----
    // (a + bj) * (c + dj)
    // = a*c - b*d +  (a*d + b*c)j
    float mr = ar*fracr - ai*fraci;
    float mi = ar*fraci + ai*fracr;
    
    // z - a * (f(z) / f'(z))
    *rr = zr - mr;
    *ri = zi - mi;
}


/**
 * f(z) = sin(z)
 * f'(z) = cos(z)
 */
void
fractal_newton_sin(float * rr, float * ri,
                   const float zr, const float zi,
                   const float ar, const float ai)
{
    // ---- f(z) ----
    // sin(a + bj) = sin(a)*cosh(b) + cos(a)*sinh(b)*j
    float fr = sin(zr)*cosh(zi);
    float fi = cos(zr)*sinh(zi);

    // ---- f'(z) ----
    // cos(a + bj) = cos(a)*cosh(b) - sin(a)*sinh(b)*j
    float dfr = cos(zr)*cosh(zi);
    float dfi = -sin(zr)*sinh(zi);

    // ---- f(z) / f'(z) ----
    float denom = dfr*dfr + dfi*dfi;
    float fracr = (fr*dfr + fi*dfi) / denom;
    float fraci = (fi*dfr - fr*dfi) / denom;

    // ---- a * (f(z) / f'(z)) ----
    float mr = ar*fracr - ai*fraci;
    float mi = ar*fraci + ai*fracr;

    // z - a * (f(z) / f'(z))
    *rr = zr - mr;
    *ri = zi - mi;
}


/**
 * f(z) = sin(z)-1
 * f'(z) = cos(z)
 */
void
fractal_newton_sin1(float * rr, float * ri,
                    const float zr, const float zi,
                    const float ar, const float ai)
{
    // ---- f(z) ----
    // sin(a + bj) = sin(a)*cosh(b) + cos(a)*sinh(b)*j
    float fr = sin(zr)*cosh(zi) - 1;
    float fi = cos(zr)*sinh(zi);

    // ---- f'(z) ----
    // cos(a + bj) = cos(a)*cosh(b) - sin(a)*sinh(b)*j
    float dfr = cos(zr)*cosh(zi);
    float dfi = -sin(zr)*sinh(zi);

    // ---- f(z) / f'(z) ----
    float denom = dfr*dfr + dfi*dfi;
    float fracr = (fr*dfr + fi*dfi) / denom;
    float fraci = (fi*dfr - fr*dfi) / denom;

    // ---- a * (f(z) / f'(z)) ----
    float mr = ar*fracr - ai*fraci;
    float mi = ar*fraci + ai*fracr;

    // z - a * (f(z) / f'(z))
    *rr = zr - mr;
    *ri = zi - mi;
}
