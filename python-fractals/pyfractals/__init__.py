import errno
import os
import platform
import sys
from ctypes import (CDLL, POINTER, Structure, byref, c_bool, c_char_p, c_float,
                    c_int, c_void_p, cast, create_string_buffer, sizeof)
from typing import List, Tuple

from .datatypes import *

p = os.path.dirname(os.path.abspath(__file__)) + "/resources"
os.environ["PATH"] = p + os.pathsep + os.environ["PATH"]
lib = CDLL(os.path.join(p, f"libfractal_{platform.system()}.dll"))

try:
    lib_cuda = CDLL(os.path.join(p, f"libcudafractals_{platform.system()}.dll"))
    cuda = True
except Exception:
    cuda = False


def wrap_lib_function(fname, argtypes: List = [], restype=None, cuda=False):
    func = getattr(lib if not cuda else lib_cuda, fname)
    if not func:
        return None
    func.argtypes = argtypes
    func.restype = restype
    return func


# HCMATRIX fractal_cmatrix_create(int ROWS, int COLS);
_fractal_cmatrix_create_wrapped = wrap_lib_function(
    "fractal_cmatrix_create",
    argtypes = [c_int, c_int],
    restype  = HCMATRIX
)

# HCMATRIX fractal_cmatrix_reshape(HCMATRIX hCmatrix, int ROWS_new, int COLS_new);
_fractal_cmatrix_reshape_wrapped = wrap_lib_function(
    "fractal_cmatrix_reshape",
    argtypes = [HCMATRIX, c_int, c_int],
    restype  = HCMATRIX
)

# void fractal_cmatrix_free(HCMATRIX hCmatrix);
_fractal_cmatrix_free_wrapped = wrap_lib_function(
    "fractal_cmatrix_free",
    argtypes = [HCMATRIX],
    restype  = None
)

# float * fractal_cmatrix_value(HCMATRIX hCmatrix, int row, int col);
_fractal_cmatrix_value_wrapped = wrap_lib_function(
    "fractal_cmatrix_value",
    argtypes = [HCMATRIX, c_int, c_int],
    restype  = c_float_p
)

# void fractal_get_colors(HCMATRIX hCmatrix, struct FractalProperties * fp);
_fractal_get_colors_wrapped = wrap_lib_function(
    "fractal_get_colors",
    argtypes = [HCMATRIX, POINTER(FractalProperties)],
    restype  = None
)

# void fractal_get_colors_th(HCMATRIX hCmatrix, struct FractalProperties * fp, int num_threads);
_fractal_get_colors_th_wrapped = wrap_lib_function(
    "fractal_get_colors_th",
    argtypes = [HCMATRIX, POINTER(FractalProperties), c_int],
    restype  = None
)

# void fractal_avxf_get_colors(HCMATRIX hCmatrix, struct FractalProperties * fp);
_fractal_avxf_get_colors_wrapped = wrap_lib_function(
    "fractal_avxf_get_colors",
    argtypes = [HCMATRIX, POINTER(FractalProperties)],
    restype  = None
)

# void fractal_avxf_get_colors_th(HCMATRIX hCmatrix, struct FractalProperties * fp, int num_threads);
_fractal_avxf_get_colors_th_wrapped = wrap_lib_function(
    "fractal_avxf_get_colors_th",
    argtypes = [HCMATRIX, POINTER(FractalProperties), c_int],
    restype  = None
)

# float fractal_cmatrix_max(HCMATRIX hCmatrix);
_fractal_cmatrix_max_wrapped = wrap_lib_function(
    "fractal_cmatrix_max",
    argtypes = [HCMATRIX],
    restype  = c_float
)

# void fractal_cmatrix_save(HCMATRIX hCmatrix, const char * filename, enum FC_Color color);
_fractal_cmatrix_save_wrapped = wrap_lib_function(
    "fractal_cmatrix_save",
    argtypes = [HCMATRIX, c_char_p, c_int],
    restype  = None
)

# void fractal_image_save(int * image, int width, int height, const char * filename, enum FC_Color color);
_fractal_image_save_wrapped = wrap_lib_function(
    "fractal_image_save",
    argtypes = [c_int_p, c_int, c_int, c_char_p, c_int],
    restype  = None
)

# void fractal_value_to_color(float * r, float * g, float * b, int value, enum FC_Color color);
_fractal_value_to_color_wrapped = wrap_lib_function(
    "fractal_value_to_color",
    argtypes = [c_float_p, c_float_p, c_float_p, c_int, c_int],
    restype  = None
)

# int * fractal_image_create(int ROWS, int COLS);
_fractal_image_create_wrapped = wrap_lib_function(
    "fractal_image_create",
    argtypes = [c_int, c_int],
    restype  = c_int_p
)

# void fractal_image_free(int * image);
_fractal_image_free_wrapped = wrap_lib_function(
    "fractal_image_free",
    argtypes = [c_int_p],
    restype  = None
)

# bool fractal_cuda_init(int width, int height);
if cuda:
    _fractal_cuda_init_wrapped = wrap_lib_function(
        "fractal_cuda_init",
        argtypes = [c_int, c_int],
        restype  = c_bool,
        cuda     = cuda
    )

    # void fractal_cuda_clean();
    _fractal_cuda_clean_wrapped = wrap_lib_function(
        "fractal_cuda_clean",
        argtypes = None,
        restype  = None,
        cuda     = cuda
    )

    # void fractal_cuda_get_colors(int * image, struct FractalProperties * fp);
    _fractal_cuda_get_colors_wrapped = wrap_lib_function(
        "fractal_cuda_get_colors",
        argtypes = [c_int_p, POINTER(FractalProperties)],
        restype  = None,
        cuda     = cuda
    )



def fractal_cmatrix_create(rows: int, cols: int) -> HCMATRIX:
    """
    Create a color matrix
    """
    return _fractal_cmatrix_create_wrapped(c_int(rows), c_int(cols))

def fractal_cmatrix_reshape(hCmatrix: HCMATRIX, rows_new: int, cols_new: int) -> HCMATRIX:
    """
    Reshape a color matrix
    """
    return _fractal_cmatrix_reshape_wrapped(hCmatrix, c_int(rows_new), c_int(cols_new))

def fractal_cmatrix_free(hCmatrix: HCMATRIX) -> None:
    """
    Free a color matrix
    """
    return _fractal_cmatrix_free_wrapped(hCmatrix)
    
def fractal_cmatrix_value(hCmatrix: HCMATRIX, row: int, col: int) -> float:
    """
    Get a matrix value
    """
    f_p = _fractal_cmatrix_value_wrapped(hCmatrix, c_int(row), c_int(col))
    return cast(f_p, c_float_p).contents.value

def fractal_get_colors(hCmatrix: HCMATRIX, fp: FractalProperties) -> None:
    """
    Get all fractal colors
    """
    return _fractal_get_colors_wrapped(hCmatrix, byref(fp))

def fractal_get_colors_th(hCmatrix: HCMATRIX, fp: FractalProperties, num_threads: int) -> None:
    """
    Get fractal colors with threading
    """
    return _fractal_get_colors_th_wrapped(hCmatrix, byref(fp), c_int(num_threads))

def fractal_avxf_get_colors(hCmatrix: HCMATRIX, fp: FractalProperties) -> None:
    """
    Get fractal colors with AVX2
    """
    return _fractal_avxf_get_colors_wrapped(hCmatrix, byref(fp))

def fractal_avxf_get_colors_th(hCmatrix: HCMATRIX, fp: FractalProperties, num_threads: int) -> None:
    """
    Get fractal colors with AVX2 and threading
    """
    return _fractal_avxf_get_colors_th_wrapped(hCmatrix, byref(fp), c_int(num_threads))

def fractal_cmatrix_max(hCmatrix: HCMATRIX) -> float:
    """
    Get the maximum color value
    """
    return _fractal_cmatrix_max_wrapped(hCmatrix)

def fractal_cmatrix_save(hCmatrix: HCMATRIX, filename: str, color: Color) -> None:
    """
    Save a color matrix as png
    """
    return _fractal_cmatrix_save_wrapped(hCmatrix, filename.encode('utf-8'), c_int(color.value))

def fractal_image_save(image: c_int_p, width: int, height: int, filename: str, color: Color) -> None:
    """
    Save an image array as png
    """
    return _fractal_image_save_wrapped(image, c_int(width), c_int(height), filename.encode('utf-8'), c_int(color.value))

def fractal_value_to_color(value: int, color: Color) -> Tuple[float, float, float]:
    """
    Convert a cmatrix value to rgb
    """
    r = c_float(0.0)
    g = c_float(0.0)
    b = c_float(0.0)
    _fractal_value_to_color_wrapped(byref(r), byref(g), byref(b), c_int(int(value)), c_int(color.value))
    return (r.value, g.value, b.value)

def fractal_image_create(width: int, height: int) -> c_int_p:
    """
    Create an image array
    """
    return _fractal_image_create_wrapped(c_int(height), c_int(width))

def fractal_image_free(image: c_int_p) -> None:
    """
    Free an image array
    """
    return _fractal_image_free_wrapped(image)

def fractal_cuda_init(width: int, height: int) -> bool:
    """
    Allocate the required memory on the CUDA device
    """
    return _fractal_cuda_init_wrapped(c_int(width), c_int(height))

def fractal_cuda_clean() -> None:
    """
    Free the allocated memory
    """
    return _fractal_cuda_clean_wrapped()

def fractal_cuda_get_colors(image: c_int_p, fp: FractalProperties) -> None:
    """
    Do the color computation
    """
    return _fractal_cuda_get_colors_wrapped(image, byref(fp))
