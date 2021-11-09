#ifndef __FRACTAL_CUDA_H__
#define __FRACTAL_CUDA_H__


#include <stdbool.h>
#include "fractal_color.h"

#ifdef __linux__
#define FCUDAAPI
#else
#define FCUDAAPI __declspec(dllexport)
#endif


#if defined(__cplusplus)
extern "C" {            // Prevents name mangling of functions
#endif


/**
 * @brief Allocate memory on the cuda device.
 * 
 * @param width 
 * @param height 
 * @return true 
 * @return false 
 */
bool FCUDAAPI fractal_cuda_init(int width, int height);

/**
 * @brief Free memory on the cuda device.
 * 
 */
void FCUDAAPI fractal_cuda_clean();

/**
 * @brief Create an image array.
 * 
 * @param width 
 * @param height 
 * @return uint8_t* 
 */
uint8_t FCUDAAPI *  fractal_cuda_image_create(int width, int height);

/**
 * @brief Free an image array.
 * 
 * @param image 
 */
void FCUDAAPI fractal_cuda_image_free(uint8_t * image);

/**
 * @brief Compute the fractal colors with cuda.
 * 
 * @param image 
 * @param fp 
 */
void FCUDAAPI fractal_cuda_get_colors(uint8_t * image, struct FractalProperties * fp);

/**
 * @brief Save an image array as png.
 * 
 * @param image 
 * @param width 
 * @param height 
 * @param filename 
 */
void FCUDAAPI fractal_cuda_image_save(uint8_t * image, int width, int height, const char * filename);


#if defined(__cplusplus)
}
#endif

#endif // __FRACTAL_CUDA_H__
