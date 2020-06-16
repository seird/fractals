"""
This is a wrapper for a design created by 'Qt 5 Designer'

To use:
    - create a design in 'Qt 5 Designer' and save as 'designs/main.ui'
    - $ pyuic5 -x designs/main.ui -o designs/main_design.py

Note that any code in qt_main_design.py will be overwritten upon recompiling
"""

import sys

from PyQt5 import QtCore, QtGui, QtWidgets

import pyfractals as pf

from .designs.main_design import Ui_MainWindow

ROWS = 1000
COLS = 1000

C_REAL_RANGE = 4
C_IMAG_RANGE = 4

X_RANGE = 4
Y_RANGE = 4


class Window(QtWidgets.QMainWindow, Ui_MainWindow):
    def __init__(self):
        super(Window, self).__init__()
        self.setupUi(self)

        self.hCmatrix = pf.fractal_cmatrix_create(ROWS, COLS)

        self.initUI()
        self.reset()
        self.link_callbacks()
        self.show()
        
    def initUI(self):
        self.setWindowTitle("Fractals")

        self.radio_fractal_z2.setChecked(True)
        self.radio_color_ultra.setChecked(True)
        self.radio_mode_julia.setChecked(True)
        self.fractal = pf.Fractal.Z2
        self.color = pf.Color.ULTRA
        self.mode = pf.Mode.JULIA

        self.label_display.resize(ROWS, COLS)
        #self.resize(0,0)
        self.setFixedSize(0, 0)

    def reset(self):
        self.c_real = 0
        self.c_imag = 0
        self.x_start = -2
        self.x_end = 2
        self.y_start = -2
        self.y_end = 2
        self.iterations = 500

        self.slider_real.setValue(self.c_real)
        self.label_real.setText(f"{self.c_real:5.2f}")
        self.slider_imag.setValue(self.c_imag)
        self.label_imag.setText(f"{self.c_imag:5.2f}")
        self.spin_iterations.setValue(self.iterations)

        self.compute()

    def update_mode(self, mode: pf.Mode):
        self.mode = mode
        self.compute()

    def update_color(self, color: pf.Color):
        self.color = color
        self.compute()

    def update_fractal(self, fractal: pf.Fractal):
        self.fractal = fractal
        self.compute()

    def update_c_real(self, slider: QtWidgets.QSlider):
        self.c_real = slider.value() / (slider.maximum() - slider.minimum()) * C_REAL_RANGE
        self.label_real.setText(f"{self.c_real:5.2f}")
        if self.mode == pf.Mode.JULIA:
            self.compute()

    def update_c_imag(self, slider: QtWidgets.QSlider):
        self.c_imag = slider.value() / (slider.maximum() - slider.minimum()) * C_IMAG_RANGE
        self.label_imag.setText(f"{self.c_imag:5.2f}")
        if self.mode == pf.Mode.JULIA:
            self.compute()

    def update_iterations(self, spinbox: QtWidgets.QSpinBox):
        self.iterations = spinbox.value()
        self.compute()
        
    def compute(self):
        fractal_properties = pf.FractalProperties(
            mode           = self.mode,
            fractal        = self.fractal,
            c_real         = self.c_real,
            c_imag         = self.c_imag,
            x_start        = self.x_start,
            x_end          = self.x_end,
            y_start        = self.y_start,
            y_end          = self.y_end,
            x_size         = COLS,
            y_size         = ROWS,
            max_iterations = self.iterations,
        )
        pf.fractal_avxf_get_colors_th(self.hCmatrix, fractal_properties, 12)
        pf.fractal_cmatrix_save(self.hCmatrix, "fractal_temp.png", self.color)
        self.label_display.setPixmap(QtGui.QPixmap("fractal_temp.png"))
        
    def link_callbacks(self):
        self.radio_color_monochrome.clicked.connect(lambda: self.update_color(pf.Color.MONOCHROME))
        self.radio_color_ultra.clicked.connect(lambda: self.update_color(pf.Color.ULTRA))
        self.radio_color_tri.clicked.connect(lambda: self.update_color(pf.Color.TRI))

        self.radio_mode_julia.clicked.connect(lambda: self.update_mode(pf.Mode.JULIA))
        self.radio_mode_mandelbrot.clicked.connect(lambda: self.update_mode(pf.Mode.MANDELBROT))
        self.radio_mode_buddhabrot.clicked.connect(lambda: self.update_mode(pf.Mode.BUDDHABROT))

        self.radio_fractal_z2.clicked.connect(lambda: self.update_fractal(pf.Fractal.Z2))
        self.radio_fractal_z3.clicked.connect(lambda: self.update_fractal(pf.Fractal.Z3))
        self.radio_fractal_z4.clicked.connect(lambda: self.update_fractal(pf.Fractal.Z4))

        self.slider_real.valueChanged.connect(lambda: self.update_c_real(self.slider_real))
        self.slider_imag.valueChanged.connect(lambda: self.update_c_imag(self.slider_imag))

        self.spin_iterations.valueChanged.connect(lambda: self.update_iterations(self.spin_iterations))

        self.pb_reset.clicked.connect(self.reset)

    def closeEvent(self, event):
        event.accept()


def main():
    app = QtWidgets.QApplication(sys.argv)

    with open("gui/styles/dark.css", "r") as f:
        stylesheet = f.read()
        app.setStyleSheet(stylesheet)
    
    window = Window()
    
    sys.exit(app.exec_())

if __name__ == '__main__':    
    main()
