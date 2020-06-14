import errno
import os
import platform
from ctypes import (CDLL, POINTER, Structure, byref, c_char_p, c_float, c_int,
                    c_void_p, cast)
from typing import List, Tuple


from .datatypes import *

lib_path = os.path.join(os.path.dirname(__file__), f"resources/libfractal_{platform.system()}.dll")

if not os.path.exists(lib_path):
    raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), lib_path)

lib = CDLL(lib_path)


def wrap_lib_function(fname, argtypes: List = [], restype=None):
    func = getattr(lib, fname)
    if not func:
        return None
    func.argtypes = argtypes
    func.restype = restype
    return func


# HCMATRIX fractal_cmatrix_create(int ROWS, int COLS);
fractal_cmatrix_create_wrapped = wrap_lib_function(
    "fractal_cmatrix_create",
    argtypes = [c_int, c_int],
    restype  = HCMATRIX
)

# HCMATRIX fractal_cmatrix_reshape(HCMATRIX hCmatrix, int ROWS_new, int COLS_new);
fractal_cmatrix_reshape_wrapped = wrap_lib_function(
    "fractal_cmatrix_reshape",
    argtypes = [HCMATRIX, c_int, c_int],
    restype  = HCMATRIX
)

# void fractal_cmatrix_free(HCMATRIX hCmatrix);
fractal_cmatrix_free_wrapped = wrap_lib_function(
    "fractal_cmatrix_free",
    argtypes = [HCMATRIX],
    restype  = None
)

# float * fractal_cmatrix_value(HCMATRIX hCmatrix, int row, int col);
fractal_cmatrix_value_wrapped = wrap_lib_function(
    "fractal_cmatrix_value",
    argtypes = [HCMATRIX, c_int, c_int],
    restype  = c_float_p
)

# void fractal_get_colors(HCMATRIX hCmatrix, struct FractalProperties * fp);
fractal_get_colors_wrapped = wrap_lib_function(
    "fractal_get_colors",
    argtypes = [HCMATRIX, POINTER(FractalProperties)],
    restype  = None
)

# void fractal_get_colors_th(HCMATRIX hCmatrix, struct FractalProperties * fp, int num_threads);
fractal_get_colors_th_wrapped = wrap_lib_function(
    "fractal_get_colors_th",
    argtypes = [HCMATRIX, POINTER(FractalProperties), c_int],
    restype  = None
)

# void fractal_avxf_get_colors(HCMATRIX hCmatrix, struct FractalProperties * fp);
fractal_avxf_get_colors_wrapped = wrap_lib_function(
    "fractal_avxf_get_colors",
    argtypes = [HCMATRIX, POINTER(FractalProperties)],
    restype  = None
)

# void fractal_avxf_get_colors_th(HCMATRIX hCmatrix, struct FractalProperties * fp, int num_threads);
fractal_avxf_get_colors_th_wrapped = wrap_lib_function(
    "fractal_avxf_get_colors_th",
    argtypes = [HCMATRIX, POINTER(FractalProperties), c_int],
    restype  = None
)

# float fractal_cmatrix_max(HCMATRIX hCmatrix);
fractal_cmatrix_max_wrapped = wrap_lib_function(
    "fractal_cmatrix_max",
    argtypes = [HCMATRIX],
    restype  = c_float
)

# void fractal_cmatrix_save(HCMATRIX hCmatrix, const char * filename, enum Color color);
fractal_cmatrix_save_wrapped = wrap_lib_function(
    "fractal_cmatrix_save",
    argtypes = [HCMATRIX, c_char_p, c_int],
    restype  = None
)

# void fractal_value_to_color(float * r, float * g, float * b, int value, enum Color color);
fractal_value_to_color_wrapped = wrap_lib_function(
    "fractal_value_to_color",
    argtypes = [c_float_p, c_float_p, c_float_p, c_int, c_int],
    restype  = None
)



def fractal_cmatrix_create(rows: int, cols: int) -> HCMATRIX:
    """
    Create a color matrix
    """
    return fractal_cmatrix_create_wrapped(c_int(rows), c_int(cols))

def fractal_cmatrix_reshape(hCmatrix: HCMATRIX, rows_new: int, cols_new: int) -> HCMATRIX:
    """
    Reshape a color matrix
    """
    return fractal_cmatrix_reshape_wrapped(hCmatrix, c_int(rows_new), c_int(cols_new))

def fractal_cmatrix_free(hCmatrix: HCMATRIX) -> None:
    """
    Free a color matrix
    """
    return fractal_cmatrix_free_wrapped(hCmatrix)
    
def fractal_cmatrix_value(hCmatrix: HCMATRIX, row: int, col: int) -> float:
    """
    Get a matrix value
    """
    f_p = fractal_cmatrix_value_wrapped(hCmatrix, c_int(row), c_int(col))
    return cast(f_p, c_float_p).contents.value

def fractal_get_colors(hCmatrix: HCMATRIX, fp: FractalProperties) -> None:
    """
    Get all fractal colors
    """
    return fractal_get_colors_wrapped(hCmatrix, byref(fp))

def fractal_get_colors_th(hCmatrix: HCMATRIX, fp: FractalProperties, num_threads: int) -> None:
    """
    Get fractal colors with threading
    """
    return fractal_get_colors_th_wrapped(hCmatrix, byref(fp), c_int(num_threads))

def fractal_avxf_get_colors(hCmatrix: HCMATRIX, fp: FractalProperties) -> None:
    """
    Get fractal colors with AVX2
    """
    return fractal_avxf_get_colors_wrapped(hCmatrix, byref(fp))

def fractal_avxf_get_colors_th(hCmatrix: HCMATRIX, fp: FractalProperties, num_threads: int) -> None:
    """
    Get fractal colors with AVX2 and threading
    """
    return fractal_avxf_get_colors_th_wrapped(hCmatrix, byref(fp), c_int(num_threads))

def fractal_cmatrix_max(hCmatrix: HCMATRIX) -> float:
    """
    Get the maximum color value
    """
    return fractal_cmatrix_max_wrapped(hCmatrix)

def fractal_cmatrix_save(hCmatrix: HCMATRIX, filename: bytes, color: Color) -> None:
    """
    Save a color matrix as png
    """
    return fractal_cmatrix_save_wrapped(hCmatrix, filename, c_int(color.value))

def fractal_value_to_color(value: int, color: Color) -> Tuple[float, float, float]:
    """
    Convert a cmatrix value to rgb
    """
    r = c_float(0.0)
    g = c_float(0.0)
    b = c_float(0.0)
    fractal_value_to_color_wrapped(byref(r), byref(g), byref(b), c_int(int(value)), c_int(color.value))
    return (r.value, g.value, b.value)
