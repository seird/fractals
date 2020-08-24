## Python wrapper - pyfractals

[Python wrapper](pyfractals) for [c-fractals](c-fractals)

### Requirements

- "[libfractals_Windows.dll](https://gitlab.com/kdries/opengl-fractals/builds/artifacts/master/raw/c-fractals/libfractal_Windows.dll?job=build_gui)" or "[libfractal_Linux.dll](https://gitlab.com/kdries/opengl-fractals/builds/artifacts/master/raw/c-fractals/libfractal_Linux.dll?job=build)" from [c-fractals](c-fractals)
- "libwinpthread-1.dll"


## PyQt5 GUI

[PyQt GUI](gui) that displays the different fractal modes.

[Download Windows GUI.](https://gitlab.com/kdries/opengl-fractals/builds/artifacts/master/raw/python-fractals/PyFractals.zip?job=build_gui).


### Requirements

- PyQt5
- pyfractals

### Launch

```
$ python start_gui.py
```

### Create pyinstaller executable

```
$ pyinstaller fractals.spec
```

![gui example](images/gui_example.gif)
