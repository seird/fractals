import time

import pyfractals as pf


def example_flame():
    height = 900
    width = 900

    start = time.time()

    properties = pf.FractalProperties(
        mode=pf.Mode.FLAMES,
        flame=pf.Flame(
            width=width,
            height=height,
            num_chaos_games=500000,
            chaos_game_length=100,
            supersample=3,
            gamma=2.2,
            savename="pyflame.png"
        )
    )

    pf.fractal_get_colors(None, properties)

    print("Total elapsed:", time.time() - start)


if __name__ == "__main__":
    example_flame()
