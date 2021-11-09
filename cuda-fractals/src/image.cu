#include "../include/fractal_cuda.h"


#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"


extern "C" void
fractal_cuda_image_save(uint8_t * image, int width, int height, const char * filename)
{
    stbi_write_png(filename, width, height, 3, image, 0);
}

extern "C" uint8_t *
fractal_cuda_image_create(int width, int height)
{
    return (uint8_t *) malloc(sizeof(uint8_t)*height*width*3);
}

extern "C" void
fractal_cuda_image_free(uint8_t * image)
{
    free(image);
}
