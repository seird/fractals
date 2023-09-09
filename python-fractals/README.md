# Python wrapper - pyfractals

Python wrapper for [c-fractals](../c-fractals), [cuda-fractals](../cuda-fractals) and [opencl-fractals](../opencl-fractals).


See `testpyfractals.py` for an example.


## Requirements

- Compiled [c-fractals shared library](../c-fractals) in `pyfractals/resources/libfractals_{OS}.dll`:
    - `pyfractals/resources/libfractals_Windows.dll`
    - `pyfractals/resources/libfractals_Linux.dll`
- Optionally, the [cuda](../cuda-fractals/) and [opencl](../opencl-fractals/) compiled shared library:
    - `pyfractals/resources/libcudafractals_Windows.dll`
    - `pyfractals/resources/libopenclfractals_Windows.dll`
- pthreads library in your system path:
    - "libwinpthread-1.dll" on windows
    - "libpthread.so.0" on linux


## Create a pip package

```
$ python -m build
```

## Install and use the pip package

```
$ pip install .\dist\pyfractals-0.0.1-py3-none-any.whl
$ python
>>> import pyfractals as pf
```
