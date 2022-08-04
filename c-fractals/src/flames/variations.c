#include "variations.h"
#include <math.h>
#include <stdlib.h>
#include <stdbool.h>


#define PARAM_R     sqrtf(point.x*point.x + point.y*point.y)
#define PARAM_1_R   (1/PARAM_R)
#define PARAM_R2    (point.x*point.x + point.y*point.y)
#define PARAM_1_R2  (1/PARAM_R2)
#define PARAM_THETA atan2f(point.x, point.y)
#define PARAM_PHI   atan2f(point.y, point.x)


// 0
struct Vector2
variation_linear(struct Vector2 point)
{
    return point;
}


// 1
struct Vector2
variation_sinusoidal(struct Vector2 point)
{
    return (struct Vector2) {
        sinf(point.x),
        sinf(point.y)
    };
}


// 2
struct Vector2
variation_spherical(struct Vector2 point)
{
    float _r = PARAM_1_R;
    return (struct Vector2) {
        _r * point.x,
        _r * point.y
    };
}


// 3
struct Vector2
variation_swirl(struct Vector2 point)
{
    float r2 = PARAM_R2;
    return (struct Vector2) {
        point.x*sinf(r2) - point.y*cosf(r2),
        point.x*cosf(r2) + point.y*sinf(r2)
    };
}


// 4
struct Vector2
variation_horseshoe(struct Vector2 point)
{
    float _r = PARAM_1_R;
    return (struct Vector2) {
        _r*((point.x - point.y)*(point.x + point.y)),
        2.0f*_r*point.x*point.y
    };
}


// 5
struct Vector2
variation_polar(struct Vector2 point)
{
    return (struct Vector2) {
        M_1_PI * PARAM_THETA,
        PARAM_R - 1.0f
    };
}


// 6
struct Vector2
variation_handkerchief(struct Vector2 point)
{
    float r = PARAM_R;
    float theta = PARAM_THETA;
    return (struct Vector2) {
        r*sinf(theta + r),
        r*cosf(theta - r)
    };
}


// 7
struct Vector2
variation_heart(struct Vector2 point)
{
    float r = PARAM_R;
    float theta = PARAM_THETA;
    return (struct Vector2) {
        r*sinf(theta*r),
        -r*cosf(theta*r)
    };
}


// 8
struct Vector2
variation_disc(struct Vector2 point)
{
    float theta_pi = PARAM_THETA*M_1_PI;
    float arg = M_PI*PARAM_R;
    return (struct Vector2) {
        theta_pi*sinf(arg),
        theta_pi*cosf(arg)
    };
}


// 9
struct Vector2
variation_spiral(struct Vector2 point)
{
    float theta = PARAM_THETA;
    float r = PARAM_R;
    float _r = PARAM_1_R;
    return (struct Vector2) {
        _r*(cosf(theta) + sinf(r)),
        _r*(sinf(theta) - cosf(r))
    };
}


// 10
struct Vector2
variation_hyperbolic(struct Vector2 point)
{
    float theta = PARAM_THETA;
    float r = PARAM_R;
    return (struct Vector2) {
        sinf(theta)/r,
        r*cosf(theta)
    };
}


// 11
struct Vector2
variation_diamond(struct Vector2 point)
{
    float theta = PARAM_THETA;
    float r = PARAM_R;
    return (struct Vector2) {
        sinf(theta)*cosf(r),
        cosf(theta)*sinf(r)
    };
}


// 12
struct Vector2
variation_ex(struct Vector2 point)
{
    float theta = PARAM_THETA;
    float r = PARAM_R;
    float p0 = sinf(theta + r);
    float p1 = cosf(theta - r);
    return (struct Vector2) {
        r*(p0*p0*p0 + p1*p1*p1),
        r*(p0*p0*p0 - p1*p1*p1)
    };
}


// 13
struct Vector2
variation_julia(struct Vector2 point)
{
    float theta = PARAM_THETA;
    float sqrt_r = sqrtf(PARAM_R);
    float omega = (rand() % 2) * M_PI; // 0 or pi
    return (struct Vector2) {
        sqrt_r*(cosf(0.5f*theta + omega)),
        sqrt_r*(sinf(0.5f*theta + omega))
    };
}


// 14
struct Vector2
variation_bent(struct Vector2 point)
{
    return (struct Vector2) {
        point.x >= 0 ? point.x : 2*point.x,
        point.y >= 0 ? point.y : 0.5f*point.y
    };
    
}


// 15
struct Vector2
variation_waves(struct Vector2 point)
{
    float b = 1.0f;
    float c2 = 1.0f;
    float e = 1.0f;
    float f2 = 1.0f;
    return (struct Vector2) {
        point.x + b*sinf(point.y/c2),
        point.y + e*sinf(point.x/f2)
    };
}


// 16
struct Vector2
variation_fisheye(struct Vector2 point)
{
    float factor = 2 / (1 + PARAM_R);
    return (struct Vector2) {
        factor*point.y,
        factor*point.x
    };
}


// 17
struct Vector2
variation_popcorn(struct Vector2 point)
{
    float c = 1.0f;
    float f = 1.0f;
    return (struct Vector2) {
        point.x + c*sinf(tanf(3*point.y)),
        point.y + f*sinf(tanf(3*point.x))
    };
}


// 18
struct Vector2
variation_exponential(struct Vector2 point)
{
    return (struct Vector2) {
        expf(point.x-1)*cosf(M_PI*point.y),
        expf(point.x-1)*sinf(M_PI*point.y)
    };
}


// 19
struct Vector2
variation_power(struct Vector2 point)
{
    float theta = PARAM_THETA;
    float factor = powf(PARAM_R, sinf(theta));
    return (struct Vector2) {
        factor*cosf(theta),
        factor*sinf(theta)
    };
}


// 20
struct Vector2
variation_cosine(struct Vector2 point)
{
    return (struct Vector2) {
        cosf(M_PI*point.x)*coshf(point.y),
        -sinf(M_PI*point.x)*sinhf(point.y)
    };
}


// 21
struct Vector2
variation_rings(struct Vector2 point)
{
    float theta = PARAM_THETA;
    float r = PARAM_R;
    float c2 = 1.0f;
    float factor = fmodf(r + c2, 2*c2) - c2 + r*(1 - c2);
    return (struct Vector2) {
        factor*cosf(theta),
        factor*sinf(theta)
    };
}


// 22
struct Vector2
variation_fan(struct Vector2 point)
{
    float c = 1.0f;
    float f = 1.0f;
    float t = M_PI*c*c;
    float t_2 = 0.5*t;
    float theta = PARAM_THETA;
    float r = PARAM_R;

    if (fmodf(theta+f, t) > t_2) {
        return (struct Vector2) {
            r*cosf(theta-t_2),
            r*sinf(theta-t_2)
        };
    } else {
        return (struct Vector2) {
            r*cosf(theta+t_2),
            r*sinf(theta+t_2)
        };
    }
}


// 23
struct Vector2
variation_blob(struct Vector2 point)
{
    float p1 = 1.0f; // high
    float p2 = 0.0f; // low
    float p3 = 1.0f; // waves
    float theta = PARAM_THETA;
    float r = PARAM_R;
    float factor = r*(p2 + 0.5*(p1 - p2)*(1 + sinf(theta*p3)));
    return (struct Vector2) {
        factor*cosf(theta),
        factor*sinf(theta)
    };
}


// 24
struct Vector2
variation_pdj(struct Vector2 point)
{
    float p1 = 1.0f;
    float p2 = 1.0f;
    float p3 = 1.0f;
    float p4 = 1.0f;
    return (struct Vector2) {
        sinf(p1*point.y) - cosf(p2*point.x),
        sinf(p3*point.x) - cosf(p4*point.y)
    };
}


// 25
struct Vector2
variation_fan2(struct Vector2 point)
{
    float p1 = M_PI*point.x*point.x;
    float p2 = point.y;
    float theta = PARAM_THETA;
    float r = PARAM_R;
    float t = theta + p2 - p1*truncf(2*theta*p2/p1);

    if (2*t > p1) {
        return (struct Vector2) {
            r*sinf(theta-0.5*p1),
            r*cosf(theta-0.5*p1)
        };
    } else {
        return (struct Vector2) {
            r*sinf(theta+0.5*p1),
            r*cosf(theta+0.5*p1)
        };
    }
}


// 26
struct Vector2
variation_rings2(struct Vector2 point)
{
    float p = 1.0f;
    float theta = PARAM_THETA;
    float r = PARAM_R;
    float t = r - 2*p*truncf(0.5*(r+p)/p) + r*(1 - p);
    return (struct Vector2) {
        t*sinf(theta),
        t*cosf(theta)
    };
}


// 27
struct Vector2
variation_eyefish(struct Vector2 point)
{
    float factor = 2/(PARAM_R+1);
    return (struct Vector2) {
        factor*point.x,
        factor*point.y
    };
}


// 28
struct Vector2
variation_bubble(struct Vector2 point)
{
    float r = PARAM_R;
    float factor = 4/(r*r + 4);
    return (struct Vector2) {
        factor*point.x,
        factor*point.y
    };
}


// 29
struct Vector2
variation_cylinder(struct Vector2 point)
{
    return (struct Vector2) {
        sinf(point.x),
        point.y
    };
}


// 30


// 31
struct Vector2
variation_noise(struct Vector2 point)
{
    float xsi1 = random_float(0, 1);
    float xsi2 = random_float(0, 1);
    return (struct Vector2) {
        xsi1*point.x*cosf(2*M_PI*xsi2),
        xsi1*point.y*sinf(2*M_PI*xsi2)
    };
}


// 32
// 33


// 34
struct Vector2
variation_blur(struct Vector2 point)
{
    (void) point;

    float xsi1 = random_float(0, 1);
    float xsi2 = random_float(0, 1);
    return (struct Vector2) {
        xsi1*cosf(2*M_PI*xsi2),
        xsi1*sinf(2*M_PI*xsi2)
    };
}


// 35
struct Vector2
variation_gaussian(struct Vector2 point)
{
    float xsi5 = random_float(0, 1);
    float xsi1_sum = 0;
    for (int k=0; k<4; ++k) {
        xsi1_sum += random_float(0, 1);
    }
    return (struct Vector2) {
        (xsi1_sum-2)*point.x*cosf(2*M_PI*xsi5),
        (xsi1_sum-2)*point.y*sinf(2*M_PI*xsi5)
    };
}


// 36
// 37
// 38
// 39
// 40
// 41


// 42
struct Vector2
variation_tangent(struct Vector2 point)
{
    return (struct Vector2) {
        sinf(point.x)/cosf(point.y),
        tanf(point.y)
    };
}


// 43
struct Vector2
variation_square(struct Vector2 point)
{
    (void) point;
    
    float xsi1 = random_float(0, 1);
    float xsi2 = random_float(0, 1);
    return (struct Vector2) {
        xsi1 - 0.5,
        xsi2 - 0.5
    };
}


// 44
// 45
// 46
// 47


// 48
struct Vector2
variation_cross(struct Vector2 point)
{
    float factor = 1/(point.x*point.x - point.y*point.y);
    return (struct Vector2) {
        factor*point.x,
        factor*point.y
    };
}


static variationfunc_t variationfuncs[] = {
    variation_linear,
    variation_sinusoidal,
    variation_spherical,
    variation_swirl,
    variation_horseshoe,
    variation_polar,
    variation_handkerchief,
    variation_heart,
    variation_disc,
    variation_spiral,
    variation_hyperbolic,
    variation_diamond,
    variation_ex,
    variation_julia,
    variation_bent,
    variation_waves,
    variation_fisheye,
    variation_popcorn,
    variation_exponential,
    variation_power,
    variation_cosine,
    variation_rings2,
    variation_fan,
    variation_blob,
    variation_pdj,
    variation_fan2,
    variation_rings,
    variation_eyefish,
    variation_bubble,
    variation_cylinder,
    variation_noise,
    variation_blur,
    variation_gaussian,
    variation_tangent,
    variation_square,
    variation_cross
};


variationfunc_t
flames_variationfunc_get(int variation)
{
    return variationfuncs[variation % (sizeof(variationfuncs)/sizeof(variationfuncs[0]))];
}


variationfunc_t
flames_variationfunc_get_random()
{
    return variationfuncs[rand() % (sizeof(variationfuncs)/sizeof(variationfuncs[0]))];
}


struct RGB
flames_variationcolor_get(int variation)
{
    static struct RGB colors[6];
    static bool first_run = true;
    if (first_run) {
        colors[0] = (struct RGB){random_float(0.25, 1), 0,                     0                    };
        colors[1] = (struct RGB){0,                     random_float(0.25, 1), 0                    };
        colors[2] = (struct RGB){0,                     0,                     random_float(0.25, 1)};
        colors[3] = (struct RGB){random_float(0.25, 1), random_float(0.25, 1), 0                    };
        colors[4] = (struct RGB){random_float(0.25, 1), 0,                     random_float(0.25, 1)};
        colors[5] = (struct RGB){0,                     random_float(0.25, 1), random_float(0.25, 1)};
    }
    first_run = false;

    return colors[variation % (sizeof(colors) / sizeof(colors[0]))];
}


int
flames_num_variations()
{
    return sizeof(variationfuncs)/sizeof(variationfuncs[0]);
}


float
random_float(float low, float high)
{
    return low + (high - low) * ((float)rand()/RAND_MAX);
}


struct Vector2
random_point(float low, float high)
{
    return (struct Vector2) {
        random_float(low, high),
        random_float(low, high),
    };
}


struct RGB
random_color()
{
    return (struct RGB) {
        random_float(0, 1),
        random_float(0, 1),
        random_float(0, 1)
    };
}


struct RGB
add_colors(struct RGB c1, struct RGB c2)
{
    return (struct RGB) {c1.r + c2.r, c1.g + c2.g, c1.b + c2.b};
}


struct RGB
scale_color(struct RGB c, float factor)
{
    return (struct RGB) {factor*c.r, factor*c.g, factor*c.b};
}
