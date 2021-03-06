import math
import time

import pyfractals as pf


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


if __name__ == "__main__":
    example_cuda()
