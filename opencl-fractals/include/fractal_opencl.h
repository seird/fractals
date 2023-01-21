#ifndef __FRACTAL_OPENCL_H__
#define __FRACTAL_OPENCL_H__


#include <stdbool.h>
#include "fractal_color.h"


/**
 * @brief Set up the the opencl environment.
 * 
 * @param width 
 * @param height 
 * @return true 
 * @return false 
 */
bool fractal_opencl_init(int width, int height);

/**
 * @brief Clean up the opencl environment.
 * 
 */
void fractal_opencl_clean();

/**
 * @brief Create an image array.
 * 
 * @param width 
 * @param height 
 * @return uint8_t* 
 */
uint8_t * fractal_opencl_image_create(int width, int height);

/**
 * @brief Free an image array.
 * 
 * @param image 
 */
void fractal_opencl_image_free(uint8_t * image);

/**
 * @brief Compute the fractal colors with opencl.
 * 
 * @param image 
 * @param fp 
 */
void fractal_opencl_get_colors(uint8_t * image, struct FractalProperties * fp);

/**
 * @brief Save an image array as png.
 * 
 * @param image 
 * @param width 
 * @param height 
 * @param filename 
 */
void fractal_opencl_image_save(uint8_t * image, int width, int height, const char * filename);


#endif // __FRACTAL_OPENCL_H__
