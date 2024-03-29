import os
import platform
from ctypes import (CDLL, POINTER, byref, c_bool, c_char_p, c_float, c_int, cast)
from typing import List, Tuple

from .datatypes import *

p = os.path.dirname(os.path.abspath(__file__)) + "/resources"
os.environ["PATH"] = p + os.pathsep + os.environ["PATH"]
_lib = CDLL(os.path.join(p, f"libfractal_{platform.system()}.dll"))

try:
    _lib_cuda = CDLL(os.path.join(p, f"libcudafractals_{platform.system()}.dll"))
    cuda = True
except Exception:
    cuda = False

try:
    _lib_opencl = CDLL(os.path.join(p, f"libopenclfractals_{platform.system()}.dll"))
    opencl = True
except Exception:
    opencl = False


def wrap_lib_function(fname, argtypes: List = [], restype=None, lib=None):
    lib = _lib if lib is None else lib
    func = getattr(lib, fname, None)
    if not func:
        return None
    func.argtypes = argtypes
    func.restype = restype
    return func


# HCMATRIX fractal_cmatrix_create(int height, int width);
_fractal_cmatrix_create_wrapped = wrap_lib_function(
    "fractal_cmatrix_create",
    argtypes = [c_int, c_int],
    restype  = HCMATRIX
)

# HCMATRIX fractal_cmatrix_reshape(HCMATRIX hCmatrix, int height_new, int width_new);
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

# float * fractal_cmatrix_value(HCMATRIX hCmatrix, int h, int w);
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

# void fractal_value_to_color(uint8_t * r, uint8_t * g, uint8_t * b, int value, enum FC_Color color);
_fractal_value_to_color_wrapped = wrap_lib_function(
    "fractal_value_to_color",
    argtypes = [c_uint8_p, c_uint8_p, c_uint8_p, c_int, c_int],
    restype  = None
)

# bool fractal_cuda_init(int width, int height);
if cuda:
    _fractal_cuda_init_wrapped = wrap_lib_function(
        "fractal_cuda_init",
        argtypes = [c_int, c_int],
        restype  = c_bool,
        lib      = _lib_cuda
    )

    # void fractal_cuda_clean();
    _fractal_cuda_clean_wrapped = wrap_lib_function(
        "fractal_cuda_clean",
        argtypes = None,
        restype  = None,
        lib      = _lib_cuda
    )

    # int * fractal_cuda_image_create(int width, int height);
    _fractal_cuda_image_create_wrapped = wrap_lib_function(
        "fractal_cuda_image_create",
        argtypes = [c_int, c_int],
        restype  = c_uint8_p,
        lib      = _lib_cuda
    )

    # void fractal_cuda_image_free(uint8_t * image);
    _fractal_cuda_image_free_wrapped = wrap_lib_function(
        "fractal_cuda_image_free",
        argtypes = [c_uint8_p],
        restype  = None,
        lib      = _lib_cuda
    )

    # void fractal_cuda_image_save(uint8_t * image, int width, int height, const char * filename);
    _fractal_cuda_image_save_wrapped = wrap_lib_function(
        "fractal_cuda_image_save",
        argtypes = [c_uint8_p, c_int, c_int, c_char_p],
        restype  = None,
        lib      = _lib_cuda
    )

    # void fractal_cuda_get_colors(uint8_t * image, struct FractalProperties * fp);
    _fractal_cuda_get_colors_wrapped = wrap_lib_function(
        "fractal_cuda_get_colors",
        argtypes = [c_uint8_p, POINTER(FractalProperties)],
        restype  = None,
        lib      = _lib_cuda
    )


if opencl:
    _fractal_opencl_init_wrapped = wrap_lib_function(
        "fractal_opencl_init",
        argtypes = [c_int, c_int],
        restype  = c_bool,
        lib      = _lib_opencl
    )

    # void fractal_opencl_clean();
    _fractal_opencl_clean_wrapped = wrap_lib_function(
        "fractal_opencl_clean",
        argtypes = None,
        restype  = None,
        lib      = _lib_opencl
    )

    # int * fractal_opencl_image_create(int width, int height);
    _fractal_opencl_image_create_wrapped = wrap_lib_function(
        "fractal_opencl_image_create",
        argtypes = [c_int, c_int],
        restype  = c_uint8_p,
        lib      = _lib_opencl
    )

    # void fractal_opencl_image_free(uint8_t * image);
    _fractal_opencl_image_free_wrapped = wrap_lib_function(
        "fractal_opencl_image_free",
        argtypes = [c_uint8_p],
        restype  = None,
        lib      = _lib_opencl
    )

    # void fractal_opencl_image_save(uint8_t * image, int width, int height, const char * filename);
    _fractal_opencl_image_save_wrapped = wrap_lib_function(
        "fractal_opencl_image_save",
        argtypes = [c_uint8_p, c_int, c_int, c_char_p],
        restype  = None,
        lib      = _lib_opencl
    )

    # void fractal_opencl_get_colors(uint8_t * image, struct FractalProperties * fp);
    _fractal_opencl_get_colors_wrapped = wrap_lib_function(
        "fractal_opencl_get_colors",
        argtypes = [c_uint8_p, POINTER(FractalProperties)],
        restype  = None,
        lib      = _lib_opencl
    )


def fractal_cmatrix_create(height: int, width: int) -> HCMATRIX:
    """
    Create a color matrix
    """
    return _fractal_cmatrix_create_wrapped(c_int(height), c_int(width))

def fractal_cmatrix_reshape(hCmatrix: HCMATRIX, height_new: int, width_new: int) -> HCMATRIX:
    """
    Reshape a color matrix
    """
    return _fractal_cmatrix_reshape_wrapped(hCmatrix, c_int(height_new), c_int(width_new))

def fractal_cmatrix_free(hCmatrix: HCMATRIX) -> None:
    """
    Free a color matrix
    """
    return _fractal_cmatrix_free_wrapped(hCmatrix)
    
def fractal_cmatrix_value(hCmatrix: HCMATRIX, h: int, w: int) -> float:
    """
    Get a matrix value
    """
    f_p = _fractal_cmatrix_value_wrapped(hCmatrix, c_int(h), c_int(w))
    return cast(f_p, c_float_p).contents.value

def fractal_cmatrix_value_pointer(hCmatrix: HCMATRIX, h: int, w: int) -> c_float_p:
    """
    Get a matrix value reference (pointer)
    """
    f_p = _fractal_cmatrix_value_wrapped(hCmatrix, c_int(h), c_int(w))
    return cast(f_p, c_float_p)

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

def fractal_cuda_image_save(image: c_uint8_p, width: int, height: int, filename: str) -> None:
    """
    Save an image array as png
    """
    return _fractal_cuda_image_save_wrapped(image, c_int(width), c_int(height), filename.encode('utf-8'))

def fractal_value_to_color(value: int, color: Color) -> Tuple[c_uint8, c_uint8, c_uint8]:
    """
    Convert a cmatrix value to rgb
    """
    r = c_uint8(0.0)
    g = c_uint8(0.0)
    b = c_uint8(0.0)
    _fractal_value_to_color_wrapped(byref(r), byref(g), byref(b), c_int(int(value)), c_int(color.value))
    return (r.value, g.value, b.value)

def fractal_cuda_image_create(width: int, height: int) -> c_uint8_p:
    """
    Create an image array
    """
    return _fractal_cuda_image_create_wrapped(c_int(height), c_int(width))

def fractal_cuda_image_free(image: c_uint8_p) -> None:
    """
    Free an image array
    """
    return _fractal_cuda_image_free_wrapped(image)

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

def fractal_cuda_get_colors(image: c_uint8_p, fp: FractalProperties) -> None:
    """
    Do the color computation
    """
    return _fractal_cuda_get_colors_wrapped(image, byref(fp))

def fractal_opencl_image_save(image: c_uint8_p, width: int, height: int, filename: str) -> None:
    """
    Save an image array as png
    """
    return _fractal_opencl_image_save_wrapped(image, c_int(width), c_int(height), filename.encode('utf-8'))

def fractal_opencl_image_create(width: int, height: int) -> c_uint8_p:
    """
    Create an image array
    """
    return _fractal_opencl_image_create_wrapped(c_int(height), c_int(width))

def fractal_opencl_image_free(image: c_uint8_p) -> None:
    """
    Free an image array
    """
    return _fractal_opencl_image_free_wrapped(image)

def fractal_opencl_init(width: int, height: int) -> bool:
    """
    Allocate the required memory on the CUDA device
    """
    return _fractal_opencl_init_wrapped(c_int(width), c_int(height))

def fractal_opencl_clean() -> None:
    """
    Free the allocated memory
    """
    return _fractal_opencl_clean_wrapped()

def fractal_opencl_get_colors(image: c_uint8_p, fp: FractalProperties) -> None:
    """
    Do the color computation
    """
    return _fractal_opencl_get_colors_wrapped(image, byref(fp))
