import math
import time

import pyfractals as pf


def example_avx():
    height = 1200
    width = 1200

    hCmatrix = pf.fractal_cmatrix_create(height, width)

    elapsed_total = 0
    N = 10
    counter = 0
    for i in range(N):
        start = time.time()

        c_real = 0.7885 * math.cos(counter / (2 * math.pi))
        c_imag = 0.7885 * math.sin(counter / (2 * math.pi))
        counter += 0.1

        properties = pf.FractalProperties(
            fractal=pf.Fractal.Z2, mode=pf.Mode.JULIA,
            c_real=c_real, c_imag=c_imag
        )

        #pf.fractal_avxf_get_colors(hCmatrix, properties)
        pf.fractal_avxf_get_colors_th(hCmatrix, properties, 12)

        # m = pf.fractal_cmatrix_max(hCmatrix)

        elapsed = time.time() - start
        print(f"[{i}]: {elapsed} seconds.")
        elapsed_total += elapsed

        pf.fractal_cmatrix_save(hCmatrix, f"fractal_avx_ultra_{i:03d}.png", pf.Color.ULTRA)
        
    print("Total elapsed:", elapsed_total)

    pf.fractal_cmatrix_free(hCmatrix)


if __name__ == "__main__":
    example_avx()
