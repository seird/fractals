import math
import time

import pyfractals as pf


def example_avx():
    height = 1200
    width = 1200

    hCmatrix = pf.fractal_cmatrix_create(height, width)

    elapsed_total = 0
    N = 50
    counter = 0
    for i in range(N):
        start = time.time()

        c_real = 0.7885 * math.cos(counter / (2 * math.pi))
        c_imag = 0.7885 * math.sin(counter / (2 * math.pi))
        counter += 0.1

        properties = pf.FractalProperties(
            width=width, height=height,
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


def example_cuda():
    height = 10*108
    width = 8*240

    aspect_ratio = width/height

    x_start = -2
    x_end = 2

    y_start = x_start/aspect_ratio
    y_end = x_end/aspect_ratio

    pf.fractal_cuda_init(width, height)
    cuda_image = pf.fractal_cuda_image_create(width, height)

    elapsed_total = 0
    N = 10
    counter = 10
    for i in range(N):
        start = time.time()

        c_real = 0.7885 * math.cos(counter / (2 * math.pi))
        c_imag = 0.7885 * math.sin(counter / (2 * math.pi))
        counter += 0.1

        properties = pf.FractalProperties(
            width=width, height=height,
            fractal=pf.Fractal.Z2, mode=pf.Mode.JULIA,
            c_real=c_real, c_imag=c_imag,
            x_start=x_start, x_end=x_end,
            y_start=y_start, y_end=y_end,
            color=pf.Color.JET
        )

        pf.fractal_cuda_get_colors(cuda_image, properties)

        elapsed = time.time() - start
        print(f"[{i}]: {elapsed} seconds.")
        elapsed_total += elapsed

        pf.fractal_cuda_image_save(cuda_image, width, height, f"fractal_cuda_ultra_{i:03d}.png")

    print("Total elapsed:", elapsed_total)

    pf.fractal_cuda_image_free(cuda_image)
    pf.fractal_cuda_clean()    


def example_lyapunov():
    height = 2000
    width = 2000

    pf.fractal_cuda_init(width, height)
    cuda_image = pf.fractal_cuda_image_create(width, height)

    elapsed_total = 0

    for sequence in ["AB", "AAB", "ABB", "ABABA", "AABAB"]:
        for max_iterations in [1, 5, 10, 100, 1_000, 10_000]: #range(1, 10000, 10):
            start = time.time()

            properties = pf.FractalProperties(
                width=width, height=height,
                mode=pf.Mode.LYAPUNOV,
                x_start=2, x_end=4,
                y_start=2, y_end=4,
                max_iterations=max_iterations,
                lyapunov_sequence=sequence,
                color=pf.Color.TRI
            )

            pf.fractal_cuda_get_colors(cuda_image, properties)

            elapsed = time.time() - start
            print(f"{sequence:5s} - {max_iterations:6d} - {elapsed} seconds.")
            elapsed_total += elapsed

            pf.fractal_cuda_image_save(cuda_image, width, height, f"fractal_cuda_ultra_lyapunov_{sequence}_{max_iterations:010d}.png")


    print("Total elapsed:", elapsed_total)

    pf.fractal_cuda_image_free(cuda_image)
    pf.fractal_cuda_clean()    


if __name__ == "__main__":
    example_avx()
    example_cuda()
    example_lyapunov()
