# Generate a Mandelbrot mosaic consisting of small Julia fractals.


from ctypes import c_float, memmove, sizeof

import pyfractals as pf


def julia_mandelbrot_photomosaic(
    savename: str,
    color: pf.Color,
    fractal: pf.Fractal,
    height_element: int, width_element: int,
    n_height: int, n_width: int,
    c_real_start: float, c_real_end: float, c_imag_start: float, c_imag_end: float,
):
    """create a mandelbrot mosaic made of julia fractals"""
    
    hmosaic = pf.fractal_cmatrix_create(n_height*height_element, n_width*width_element)
    helement = pf.fractal_cmatrix_create(height_element, width_element)

    R = 4.0; # choose R outside of the visible range in the complex plane to avoid black outer edges in the julia fractals

    aspect_ratio = width_element/height_element

    x_start = -2
    x_end   = 2

    y_start = x_start / aspect_ratio
    y_end = x_end / aspect_ratio

    fp = pf.FractalProperties(
        x_start = x_start,
        x_end = x_end,
        y_start = y_start,
        y_end = y_end,
        fractal = fractal,
        mode = pf.Mode.JULIA,
        R = R,
        max_iterations = 500,
    )

    for i in range(n_height):
        fp.c_imag = c_imag_start + (i+0.5)*(c_imag_end - c_imag_start)/n_height
        for j in range(n_width):
            fp.c_real = c_real_start + (j+0.5)*(c_real_end - c_real_start)/n_width

            pf.fractal_avxf_get_colors(helement, fp)
            
            # copy the julia fractal into the mosaic
            for h in range(i*height_element, (i+1)*height_element):
                start_mosaic = pf.fractal_cmatrix_value_pointer(hmosaic, h, j*width_element)
                start_element = pf.fractal_cmatrix_value_pointer(helement, h-i*height_element, 0)
                
                memmove(start_mosaic, start_element, sizeof(c_float)*width_element)

    pf.fractal_cmatrix_save(hmosaic, savename, color)

    pf.fractal_cmatrix_free(hmosaic)
    pf.fractal_cmatrix_free(helement)


def example_julia_mandelbrot_mosaic():
    julia_mandelbrot_photomosaic(
        "mosaic.png", pf.Color.ULTRA, pf.Fractal.Z2,
        32, 32,              # julia fractal dimension (multiple of 8 for avx)
        100, 100,            # number of julia fractals that will be placed into a mosaic
        -2.0, 0.8, -1.4, 1.4 # range in the complex plane to draw the mandelbrot set
    )


if __name__ == "__main__":
    example_julia_mandelbrot_mosaic()
