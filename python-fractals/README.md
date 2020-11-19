# Python wrapper - pyfractals

[Python wrapper](pyfractals) for [c-fractals](c-fractals)


See `testpyfractals.py` for an example.


### Requirements

- Compiled [c-fractals library](c-fractals) in `pyfractals/resources/libfractals_{OS}.dll`:
    - `pyfractals/resources/libfractals_Windows.dll`
    - `pyfractals/resources/libfractals_Linux.dll`
- pthreads library in your system path:
    - "libwinpthread-1.dll" on windows
    - "libpthread.so.0" on linux



# PyQt5 GUI

[PyQt GUI](gui) that displays the different fractal modes.


### Requirements

- pyfractals (wrapper)
- Python 3.7.9 (tested)
- PyQt5

### Launch

```
$ python start_gui.py
```

### Create pyinstaller executable

```
$ pyinstaller fractals.spec
```

<p float="left">
  <img src="images/gui_example.gif" width="800" />
</p>

