# Raylib fractals


Toy example using raylib with zooming and panning.


## Getting started

- Install [raylib](https://github.com/raysan5/raylib/releases) (dynamic)
- gcc, make
- `libfractal.so`, compiled shared library from [c-fractals](../c-fractals)
    - `$ sudo cp libfractal.so /usr/local/lib/`
    - `$ LD_LIBRARY_PATH=/usr/local/lib`
    - `$ export LD_LIBRARY_PATH`
- `$ make run` to compile and run the example in src/main.c


![](images/example.png)
