from ctypes import (POINTER, Structure, c_char_p, c_float, c_int,
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
    FLAMES     = 3


class Color(Enum):
    ULTRA      = 0
    MONOCHROME = 1
    TRI        = 2
    JET        = 3
    LAVENDER   = 4
    BINARY     = 5


class Flame(Structure):
    _fields_ = [
        ("width"            , c_int),
        ("height"           , c_int),
        ("num_chaos_games"  , c_int),
        ("chaos_game_length", c_int),
        ("supersample"      , c_int),
        ("gamma"            , c_float),
        ("savename"         , c_char_p),
    ]

    def __init__(self,
                 width            : int,
                 height           : int,
                 num_chaos_games  : Optional[int]     = 500000,
                 chaos_game_length: Optional[int]     = 100,
                 supersample      : Optional[int]     = 3,
                 gamma            : Optional[float]   = 2.2,
                 savename         : Optional[str]     = "flame.png"):
        self.width             = width
        self.height            = height
        self.num_chaos_games   = c_int(num_chaos_games)
        self.chaos_game_length = c_int(chaos_game_length)
        self.supersample       = c_int(supersample)
        self.gamma             = c_float(gamma)
        self.savename          = savename.encode("utf8")


class FractalProperties(Structure):
    """
    FractalProperties struct
    """
    _fields_ = [
        ("x_start"          , c_float),
        ("x_end"            , c_float),
        ("y_start"          , c_float),
        ("y_end"            , c_float),
        ("frac"             , c_int),
        ("mode"             , c_int),
        ("color"            , c_int),
        ("c_real"           , c_float),
        ("c_imag"           , c_float),
        ("R"                , c_float),
        ("max_iterations"   , c_int),
        ("lyapunov_sequence", c_char_p),
        ("sequence_length"  , c_size_t),
        ("flame"            , Flame)
    ]

    def __init__(self,
                 x_start          : Optional[float]   = -2.0,
                 x_end            : Optional[float]   = 2.0,
                 y_start          : Optional[float]   = -2.0,
                 y_end            : Optional[float]   = 2.0,
                 fractal          : Optional[Fractal] = Fractal.Z2,
                 mode             : Optional[Mode]    = Mode.JULIA,
                 color            : Optional[Color]   = Color.ULTRA,
                 c_real           : Optional[float]   = -0.7835,
                 c_imag           : Optional[float]   = -0.2321,
                 R                : Optional[float]   = 2,
                 max_iterations   : Optional[int]     = 1000,
                 lyapunov_sequence: Optional[str]     = "AABAB",
                 flame            : Optional[Flame]   = Flame(0, 0)):
        self.x_start           = c_float(x_start)
        self.x_end             = c_float(x_end)
        self.y_start           = c_float(y_start)
        self.y_end             = c_float(y_end)
        self.frac              = c_int(fractal.value)
        self.mode              = c_int(mode.value)
        self.color             = c_int(color.value)
        self.c_real            = c_float(c_real)
        self.c_imag            = c_float(c_imag)
        self.R                 = c_float(R)
        self.max_iterations    = c_int(max_iterations)
        self.lyapunov_sequence = lyapunov_sequence.encode("utf8")
        self.sequence_length   = len(lyapunov_sequence)
        self.flame             = flame
