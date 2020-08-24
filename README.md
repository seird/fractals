[![pipeline status](https://gitlab.com/kdries/opengl-fractals/badges/master/pipeline.svg)](https://gitlab.com/kdries/opengl-fractals/commits/master)

# Fractals

- [C Fractal implementations](c-fractals)
- [OpenGL rendering](opengl-fractals)
- [Python wrapper and PyQt gui](python-fractals)
        [Download Windows GUI](https://gitlab.com/kdries/opengl-fractals/builds/artifacts/master/raw/python-fractals/PyFractals.zip?job=build).


## Performance

```
./benchmark.exe

=================================================
Benchmarking ...
        Number of runs     =                   10
        Fractal iterations =                 1000
        Number of threads  =                    6
        ROWS               =                 1000
        COLUMNS            =                 1000
        C_REAL             =            -0.788485
        C_IMAG             =             0.004913
        MODE               =                    1
        FRACTAL            =                    0

bench_default
          1.173700 seconds per run [11.737000 seconds total]

bench_threaded
          0.581400 seconds per run [5.814001 seconds total]

bench_avx
          0.086600 seconds per run [0.865999 seconds total]

bench_avx_threaded
          0.043200 seconds per run [0.431999 seconds total]
```

## Examples

![](python-fractals/images/gui_example.gif)

![](images/example_iteration_1.gif)

![](images/example_iteration_2.gif)

![](images/example_ultra.png)

![](images/example_zoom_ultra.gif)

![](images/example_rotate_ultra.gif)

![](images/example_gradients.png)
