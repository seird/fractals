#ifndef __KERNELS_H__
#define __KERNELS_H__


#include <stdint.h>
#include <stdlib.h>


#include "declarations.h"
#include "colormap.h"
#include "fractals.h"
#include "julia.h"
#include "mandelbrot.h"
#include "utils.h"


static inline void
load_kernels(cl_uint * num_sources, char *** sourcestrs, size_t ** sourcelengths)
{
    *num_sources = 6;
    *sourcestrs = (char **) malloc(sizeof(char *) * 6);
    *sourcelengths = (size_t *) malloc(sizeof(size_t) * 6);

    (*sourcestrs)[0] = (char *) src_kernels_declarations_cl;
    (*sourcelengths)[0] = src_kernels_declarations_cl_len;

    (*sourcestrs)[1] = (char *) src_kernels_colormap_cl;
    (*sourcelengths)[1] = src_kernels_colormap_cl_len;

    (*sourcestrs)[2] = (char *) src_kernels_fractals_cl;
    (*sourcelengths)[2] = src_kernels_fractals_cl_len;

    (*sourcestrs)[3] = (char *) src_kernels_julia_cl;
    (*sourcelengths)[3] = src_kernels_julia_cl_len;

    (*sourcestrs)[4] = (char *) src_kernels_mandelbrot_cl;
    (*sourcelengths)[4] = src_kernels_mandelbrot_cl_len;

    (*sourcestrs)[5] = (char *) src_kernels_utils_cl;
    (*sourcelengths)[5] = src_kernels_utils_cl_len;

}


#endif // __KERNELS_H__
