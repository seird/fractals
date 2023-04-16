import time

import pyfractals as pf


def example_lyapunov():
    height = 2000
    width = 2000

    pf.fractal_opencl_init(width, height)
    opencl_image = pf.fractal_opencl_image_create(width, height)

    elapsed_total = 0

    for sequence in ["AB", "AAB", "ABB", "ABABA", "AABAB"]:
        for max_iterations in [1, 5, 10, 100, 1_000, 10_000]: #range(1, 10000, 10):
            start = time.time()

            properties = pf.FractalProperties(
                mode=pf.Mode.LYAPUNOV,
                x_start=2, x_end=4,
                y_start=2, y_end=4,
                max_iterations=max_iterations,
                lyapunov_sequence=sequence,
                color=pf.Color.TRI
            )

            pf.fractal_opencl_get_colors(opencl_image, properties)

            elapsed = time.time() - start
            print(f"{sequence:5s} - {max_iterations:6d} - {elapsed} seconds.")
            elapsed_total += elapsed

            pf.fractal_opencl_image_save(opencl_image, width, height, f"fractal_opencl_ultra_lyapunov_{sequence}_{max_iterations:010d}.png")


    print("Total elapsed:", elapsed_total)

    pf.fractal_opencl_image_free(opencl_image)
    pf.fractal_opencl_clean()    


if __name__ == "__main__":
    example_lyapunov()
