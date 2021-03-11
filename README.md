[![pipeline status](https://gitlab.com/kdries/opengl-fractals/badges/master/pipeline.svg)](https://gitlab.com/kdries/opengl-fractals/commits/master)

# Fractals

- [C Fractal implementations](c-fractals)
- [CUDA Fractal implementation](cuda-fractals)
- [Raylib rendering](raylib-fractals)
- [OpenGL rendering](opengl-fractals)
- [Python wrapper and PyQt gui](python-fractals)


## Performance

cpu = 8700k @ 4.9GHz
gpu = 1080Ti

```
Number of runs     =                   10
Fractal iterations =                 1000
Number of threads  =                    6
ROWS               =                 1000
COLUMNS            =                 1000
C_REAL             =            -0.788485
C_IMAG             =             0.004913
MODE               =           MODE_JULIA
FRACTAL            =              FRAC_Z2
```

**Time per run**

|                 |  Linux (5.4.0-54)  |  Windows  |
|-----------------|:------------------:|:---------:|
|**Default**      | 1901 ms            | 721 ms    |
|**Threaded**     | 321  ms            | 137 ms    |
|**AVX2**         | 90   ms            | 91  ms    |
|**AVX2 Threaded**| 18   ms            | 21  ms    |
|**CUDA**         | -    ms            | 2   ms    |


## Examples


<p float="left">
  <img src="python-fractals/images/gui_example.gif" width="400" />
  <img src="raylib-fractals/images/example.gif" width="340" />
</p>

<p float="left">
  <img src="images/example_iteration_1.gif" width="400" />
  <img src="images/example_iteration_2.gif" width="400" /> 
</p>


![](images/example_ultra.png)

![](images/example_rotate_ultra.gif)
