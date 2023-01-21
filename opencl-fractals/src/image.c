#include "../include/fractal_opencl.h"
#include <stdlib.h>

#ifndef NOSTB
#define STB_IMAGE_WRITE_IMPLEMENTATION
#endif
#include "stb_image_write.h"


void
fractal_opencl_image_save(uint8_t * image, int width, int height, const char * filename)
{
    stbi_write_png(filename, width, height, 3, image, 0);
}

uint8_t *
fractal_opencl_image_create(int height, int width)
{
    return (uint8_t *) malloc(sizeof(uint8_t)*height*width*3);
}

void
fractal_opencl_image_free(uint8_t * image)
{
    free(image);
}
