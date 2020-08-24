## Python wrapper - pyfractals

[Python wrapper](pyfractals) for [c-fractals](c-fractals)

### Requirements

- "libfractals_OS.dll" from [c-fractals](c-fractals)
- "libwinpthread-1.dll"


## PyQt5 GUI

[PyQt GUI](gui) that displays the different fractal modes.

[Download Windows GUI.](https://gitlab.com/kdries/opengl-fractals/builds/artifacts/master/raw/python-fractals/PyFractals.zip?job=build).


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
