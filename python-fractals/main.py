import time

import pyfractals as pf


if __name__ == "__main__":
    ROWS = 1000
    COLS = 1000

    hCmatrix = pf.fractal_cmatrix_create(ROWS, COLS)

    elapsed_total = 0
    for i, c_real in enumerate([-1, -0.7835, -.5, -.3, 0, .3, .5, 0.7835, 1]):
        start = time.time()

        properties = pf.FractalProperties(x_size=COLS, y_size=ROWS, mode=pf.Mode.JULIA, c_real=c_real)

        #pf.fractal_avxf_get_colors(hCmatrix, properties)
        pf.fractal_avxf_get_colors_th(hCmatrix, properties, 12)

        m = pf.fractal_cmatrix_max(hCmatrix)
        print(f"max = {m:.5f}")
        print("pixel =", pf.fractal_value_to_color(m, pf.Color.MONOCHROME))

        elapsed = time.time() - start
        print(f"[{i}]: {elapsed} seconds.")
        elapsed_total += elapsed

        pf.fractal_cmatrix_save(hCmatrix, b"fractal_%d.png" %i, pf.Color.TRI)


    print("Total elapsed:", elapsed_total)

    pf.fractal_cmatrix_free(hCmatrix)
