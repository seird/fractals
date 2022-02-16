/* Render a fractal flame. */

#include "../include/fractal_color.h"
#include <time.h>


int
main(void)
{
    /* ----------- INPUT PARAMETERS ----------- */
    struct FractalProperties fp = {
        .mode = FC_MODE_FLAMES,
        .flame = {
            .width = 900,
            .height = 900,
            .num_chaos_games = 500000,
            .chaos_game_length = 100,
            .supersample = 3,
            .gamma = 2.2f,
            .savename = "flames.png"
        }
    };
    /* ---------------------------------------- */

    srand(time(NULL));

    fractal_get_colors(NULL, &fp);

	return 0;
}
