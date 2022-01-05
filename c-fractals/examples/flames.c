/* Render a fractal flame. */

#include "../include/fractal_color.h"


int
main(void)
{
    /* ----------- INPUT PARAMETERS ----------- */
    struct FractalProperties fp = {
        .width = 900,
        .height = 900,
        .mode = FC_MODE_FLAMES,
        .flame = {
            .num_chaos_games = 500000,
            .chaos_game_length = 100,
            .supersample = 3,
            .gamma = 2.2f,
            .savename = "flames.png"
        }
    };
    /* ---------------------------------------- */

    fractal_get_colors(NULL, &fp);

	return 0;
}
