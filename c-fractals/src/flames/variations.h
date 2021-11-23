#ifndef __VARIATIONS_H__
#define __VARIATIONS_H__


struct Vector2 {
    float x;
    float y;
};

struct RGB {
    float r;
    float g;
    float b;
};

typedef struct Vector2 (* variationfunc_t)(struct Vector2);


/**
 * @brief Get a variation function
 * 
 * @param variation 
 * @return variationfunc_t 
 */
variationfunc_t flames_variationfunc_get(int variation);


/**
 * @brief Get a random variation function
 * 
 * @return variationfunc_t 
 */
variationfunc_t flames_variationfunc_get_random();


/**
 * @brief Get the number of variations
 * 
 * @return int 
 */
int flames_num_variations();


struct RGB flames_variationcolor_get(int variation);


float random_float(float low, float high);
struct Vector2 random_point(float low, float high);
struct RGB random_color();
struct RGB add_colors(struct RGB c1, struct RGB c2);
struct RGB scale_color(struct RGB c, float factor);

struct Vector2 variation_linear(struct Vector2 point);
struct Vector2 variation_sinusoidal(struct Vector2 point);
struct Vector2 variation_spherical(struct Vector2 point);
struct Vector2 variation_swirl(struct Vector2 point);
struct Vector2 variation_horseshoe(struct Vector2 point);
struct Vector2 variation_polar(struct Vector2 point);
struct Vector2 variation_handkerchief(struct Vector2 point);
struct Vector2 variation_heart(struct Vector2 point);
struct Vector2 variation_disc(struct Vector2 point);
struct Vector2 variation_spiral(struct Vector2 point);
struct Vector2 variation_hyperbolic(struct Vector2 point);
struct Vector2 variation_diamond(struct Vector2 point);
struct Vector2 variation_ex(struct Vector2 point);
struct Vector2 variation_julia(struct Vector2 point);
struct Vector2 variation_bent(struct Vector2 point);
struct Vector2 variation_waves(struct Vector2 point);
struct Vector2 variation_fisheye(struct Vector2 point);
struct Vector2 variation_popcorn(struct Vector2 point);
struct Vector2 variation_exponential(struct Vector2 point);
struct Vector2 variation_power(struct Vector2 point);
struct Vector2 variation_cosine(struct Vector2 point);
struct Vector2 variation_rings2(struct Vector2 point);
struct Vector2 variation_fan(struct Vector2 point);
struct Vector2 variation_blob(struct Vector2 point);
struct Vector2 variation_pdj(struct Vector2 point);
struct Vector2 variation_fan2(struct Vector2 point);
struct Vector2 variation_rings(struct Vector2 point);
struct Vector2 variation_eyefish(struct Vector2 point);
struct Vector2 variation_bubble(struct Vector2 point);
struct Vector2 variation_cylinder(struct Vector2 point);
struct Vector2 variation_noise(struct Vector2 point);
struct Vector2 variation_tangent(struct Vector2 point);
struct Vector2 variation_cros(struct Vector2 point);


#endif // __VARIATIONS_H__
