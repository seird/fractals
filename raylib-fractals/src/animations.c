#include "animations.h"


static animationfunc_t animationfuncs[NUM_ANIMATIONS] = {
    animation_default,
    animation_iterations,
    animation_default_iterations,
};


animationfunc_t
animationfunc_get(enum Animation animation)
{
    return animationfuncs[animation % NUM_ANIMATIONS];
}


void
animation_default(struct FractalProperties * fp, int step, float speed)
{
    fp->c_real = 0.7885 * cosf((float)step / (2 * M_PI) / 20 * speed);
    fp->c_imag = 0.7885 * sinf((float)step / (2 * M_PI) / 10 * speed);
}


void
animation_iterations(struct FractalProperties * fp, int step, float speed)
{
    static int direction = 1;

    int period = 200 * (1 / fabsf(speed));
    int _step = step % period;

    int iterations_lb = 2;
    int iterations_ub = 100;

    if (direction == 1) {
        fp->max_iterations = iterations_lb + (1 - expf(5.0f*_step/period)) / (1 - expf(5.0f)) * iterations_ub;
    } else if (direction == -1) {
        fp->max_iterations = iterations_lb + (1 - expf(5.0f*(period - _step)/period)) / (1 - expf(5.0f)) * iterations_ub;
    }

    if (_step == (period-1)) {
        direction *= -1;
    }
}

void
animation_default_iterations(struct FractalProperties * fp, int step, float speed)
{
    animation_default(fp, step, speed);
    animation_iterations(fp, step, speed);
}
