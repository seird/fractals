#ifndef __ANIMATIONS_H__


#include <stdint.h>
#include <math.h>


#include "../../c-fractals/include/fractal_color.h"


enum Animation {
    ANIMATION_DEFAULT,
    ANIMATION_ITERATIONS,
    ANIMATION_ITERATIONS_LINEAR,
    ANIMATION_DEFAULT_ITERATIONS,
    ANIMATION_ESCAPE,
    NUM_ANIMATIONS
};


typedef void (* animationfunc_t)(struct FractalProperties * fp, int step, float speed);


animationfunc_t animationfunc_get(enum Animation animation);

void animation_default(struct FractalProperties * fp, int step, float speed);
void animation_iterations(struct FractalProperties * fp, int step, float speed);
void animation_iterations_linear(struct FractalProperties * fp, int step, float speed);
void animation_default_iterations(struct FractalProperties * fp, int step, float speed);
void animation_escape(struct FractalProperties * fp, int step, float speed);


#endif // __ANIMATIONS_H__
