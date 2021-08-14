# Raylib fractals


Toy example using raylib with zooming and panning.


## Getting started


### Requirements


- [c-fractals library](../c-fractals)

    ```
    cd ../c-fractals
    make static
    cp libfractal.a ../libfractal.a
    ```

- [cuda-fractals library](../cuda-fractals) (optional)

    - nvcc compiler:

        ```
        $ sudo apt -y install nvidia-cuda-toolkit
        ```

    ```
    cd ../cuda-fractals
    make lib
    cp libcudafractals.so ../libcudafractals.so
    cd ..
    ```


### Build Linux


- get raylib library

    ```
    cd raylib-fractals
    curl -o raylib.tar.gz -L https://github.com/raysan5/raylib/releases/download/3.7.0/raylib-3.7.0_linux_amd64.tar.gz
    tar -xf raylib.tar.gz
    cp raylib-*_linux_amd64/lib/libraylib.a ../libraylib.a
    ```

- build cpu version

    ```
    make build
    cp a_release.exe run_raylib
    ```

    run: `./run_raylib`


- build cuda version

    ```
    make build_cuda
    cp a_release.exe run_raylib_cuda
    ```

    before running, add ../libcudafractals.so to your PATH or:

        ```
        LD_LIBRARY_PATH=..
        export LD_LIBRARY_PATH
        ```

    and then run: `./run_raylib_cuda`


## Examples

![images/example.png](images/example.png)
