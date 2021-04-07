import math
import time

import pyfractals as pf


def example_avx():
    ROWS = 1000
    COLS = 1000

    hCmatrix = pf.fractal_cmatrix_create(ROWS, COLS)

    elapsed_total = 0
    N = 10
    counter = 0
    for i in range(N):
        start = time.time()

        c_real = 0.7885 * math.cos(counter / (2 * math.pi))
        c_imag = 0.7885 * math.sin(counter / (2 * math.pi))
        counter += 0.1

        properties = pf.FractalProperties(width=COLS, height=ROWS, fractal=pf.Fractal.Z2, mode=pf.Mode.JULIA, c_real=c_real, c_imag=c_imag)

        #pf.fractal_avxf_get_colors(hCmatrix, properties)
        pf.fractal_avxf_get_colors_th(hCmatrix, properties, 12)

        # m = pf.fractal_cmatrix_max(hCmatrix)

        elapsed = time.time() - start
        print(f"[{i}]: {elapsed} seconds.")
        elapsed_total += elapsed

        pf.fractal_cmatrix_save(hCmatrix, f"fractal_avx_ultra_{i:03d}.png", pf.Color.ULTRA)
        
    print("Total elapsed:", elapsed_total)

    pf.fractal_cmatrix_free(hCmatrix)


def example_cuda():
    ROWS = 1000
    COLS = 1000

    pf.fractal_cuda_init(COLS, ROWS)
    cuda_image = pf.fractal_image_create(COLS, ROWS)

    elapsed_total = 0
    N = 10
    counter = 100
    for i in range(N):
        start = time.time()

        c_real = 0.7885 * math.cos(counter / (2 * math.pi))
        c_imag = 0.7885 * math.sin(counter / (2 * math.pi))
        counter += 0.1

        properties = pf.FractalProperties(width=COLS, height=ROWS, fractal=pf.Fractal.Z2, mode=pf.Mode.JULIA, c_real=c_real, c_imag=c_imag)

        pf.fractal_cuda_get_colors(cuda_image, properties)

        elapsed = time.time() - start
        print(f"[{i}]: {elapsed} seconds.")
        elapsed_total += elapsed

        pf.fractal_image_save(cuda_image, COLS, ROWS, f"fractal_cuda_ultra_{i:03d}.png", pf.Color.ULTRA)

    print("Total elapsed:", elapsed_total)

    pf.fractal_image_free(cuda_image)
    pf.fractal_cuda_clean()    


def example_lyapunov():
    ROWS = 2000
    COLS = 2000

    pf.fractal_cuda_init(COLS, ROWS)
    cuda_image = pf.fractal_image_create(COLS, ROWS)

    elapsed_total = 0

    for sequence in ["AB", "AAB", "ABB", "ABABA", "AABAB"]:
        for max_iterations in [1, 5, 10, 100, 1_000, 10_000]: #range(1, 10000, 10):
            start = time.time()
            properties = pf.FractalProperties(width=COLS, height=ROWS, mode=pf.Mode.LYAPUNOV, x_start=2, x_end=4, y_start=2, y_end=4, max_iterations=max_iterations, lyapunov_sequence=sequence)

            pf.fractal_cuda_get_colors(cuda_image, properties)

            elapsed = time.time() - start
            print(f"{elapsed} seconds.")
            elapsed_total += elapsed

            pf.fractal_image_save(cuda_image, COLS, ROWS, f"fractal_cuda_ultra_lyapunov_{sequence}_{max_iterations:010d}.png", pf.Color.TRI)


    print("Total elapsed:", elapsed_total)

    pf.fractal_image_free(cuda_image)
    pf.fractal_cuda_clean()    


if __name__ == "__main__":
    example_avx()
    example_cuda()
    example_lyapunov()
