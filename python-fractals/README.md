# Python wrapper - pyfractals

Python wrapper for [c-fractals](../c-fractals).


See `testpyfractals.py` for an example.


## Requirements

- Compiled [c-fractals shared library](../c-fractals) in `pyfractals/resources/libfractals_{OS}.dll`:
    - `pyfractals/resources/libfractals_Windows.dll`
    - `pyfractals/resources/libfractals_Linux.dll`
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
