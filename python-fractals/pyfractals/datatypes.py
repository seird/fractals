from ctypes import (CDLL, POINTER, Structure, byref, c_char_p, c_float, c_int,
                    c_uint8, c_size_t, c_void_p)
from enum import Enum
from typing import Optional

HCMATRIX = c_void_p
c_int_p = POINTER(c_int)
c_float_p = POINTER(c_float)
c_uint8_p = POINTER(c_uint8)

class Fractal(Enum):
    Z2      = 0
    Z3      = 1
    Z4      = 2
    ZCONJ2  = 3
    ZCONJ3  = 4
    ZCONJ4  = 5
    ZABS2   = 6
    ZABS3   = 7
    ZABS4   = 8
    ZMAGNET = 9
    Z2_Z    = 10


class Mode(Enum):
    MANDELBROT = 0
    JULIA      = 1
    LYAPUNOV   = 2


class Color(Enum):
    ULTRA      = 0
    MONOCHROME = 1
    TRI        = 2
    JET        = 3
    LAVENDER   = 4
    BINARY     = 5


class FractalProperties(Structure):
    """
    FractalProperties struct
    """
    _fields_ = [
        ("x_start"          , c_float),
        ("x_end"            , c_float),
        ("y_start"          , c_float),
        ("y_end"            , c_float),
        ("width"            , c_int),
        ("height"           , c_int),
        ("frac"             , c_int),
        ("mode"             , c_int),
        ("c_real"           , c_float),
        ("c_imag"           , c_float),
        ("R"                , c_float),
        ("max_iterations"   , c_int),
        ("lyapunov_sequence", c_char_p),
        ("sequence_length"  , c_size_t)
    ]

    def __init__(self,
                 x_start          : Optional[float]   = -2.0,
                 x_end            : Optional[float]   = 2.0,
                 y_start          : Optional[float]   = -2.0,
                 y_end            : Optional[float]   = 2.0,
                 width            : Optional[int]     = 1000,
                 height           : Optional[int]     = 1000,
                 fractal          : Optional[Fractal] = Fractal.Z2,
                 mode             : Optional[Mode]    = Mode.JULIA,
                 c_real           : Optional[float]   = -0.7835,
                 c_imag           : Optional[float]   = -0.2321,
                 R                : Optional[float]   = 2,
                 max_iterations   : Optional[int]     = 1000,
                 lyapunov_sequence: Optional[str]     = "AABAB"):
        self.x_start           = c_float(x_start)
        self.x_end             = c_float(x_end)
        self.y_start           = c_float(y_start)
        self.y_end             = c_float(y_end)
        self.width             = c_int(width)
        self.height            = c_int(height)
        self.frac              = fractal.value
        self.mode              = mode.value
        self.c_real            = c_float(c_real)
        self.c_imag            = c_float(c_imag)
        self.R                 = c_float(R)
        self.max_iterations    = c_int(max_iterations)
        self.lyapunov_sequence = lyapunov_sequence.encode("utf8")
        self.sequence_length   = len(lyapunov_sequence)
