#ifndef __UI_STRINGS_H__
#define __UI_STRINGS_H__


static char * mode_str[] = {
    [FC_MODE_MANDELBROT] = "Mandelbrot",
    [FC_MODE_JULIA]      = "Julia",
    [FC_MODE_LYAPUNOV]   = "Lyapunov",
    [FC_MODE_FLAMES]     = "Flames",
    [FC_MODE_NEWTON]     = "Newton",
};


static char * fractal_str[] = {
    [FC_FRAC_Z2]     = "Z2",
    [FC_FRAC_Z3]     = "Z3",
    [FC_FRAC_Z4]     = "Z4",
    [FC_FRAC_ZCONJ2] = "ZCONJ2",
    [FC_FRAC_ZCONJ3] = "ZCONJ3",
    [FC_FRAC_ZCONJ4] = "ZCONJ4",
    [FC_FRAC_ZABS2]  = "ZABS2",
    [FC_FRAC_ZABS3]  = "ZABS3",
    [FC_FRAC_ZABS4]  = "ZABS4",
    [FC_FRAC_MAGNET] = "MAGNET",
    [FC_FRAC_Z2_Z]   = "Z2_Z",
#ifdef OPENCL
    [FC_FRAC_N_Z3]   = "N_Z3",
    [FC_FRAC_N_SIN]  = "N_SIN",
    [FC_FRAC_N_SIN1] = "N_SIN1",
#endif
};


static char * color_str[] = {
    [FC_COLOR_ULTRA]      = "ULTRA",
    [FC_COLOR_MONOCHROME] = "MONOCHROME",
    [FC_COLOR_TRI]        = "TRI",
    [FC_COLOR_JET]        = "JET",
    [FC_COLOR_LAVENDER]   = "LAVENDER",
    [FC_COLOR_BINARY]     = "BINARY",
#ifdef OPENCL
    [FC_COLOR_PURPLE]     = "PURPLE",
#endif
};


static char * animation_str[] = {
    [ANIMATION_DEFAULT]            = "DEFAULT",
    [ANIMATION_NEWTON]             = "NEWTON",
    [ANIMATION_ITERATIONS]         = "ITERATIONS",
    [ANIMATION_ITERATIONS_LINEAR]  = "ITERATIONS_LINEAR",
    [ANIMATION_DEFAULT_ITERATIONS] = "DEFAULT_ITERATIONS",
    [ANIMATION_ESCAPE]             = "ESCAPE",
};


#endif // __UI_STRINGS_H__
