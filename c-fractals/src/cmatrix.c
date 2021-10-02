#include "main.h"

#ifndef STATIC
#define STB_IMAGE_WRITE_IMPLEMENTATION
#endif
#include "stb_image_write.h"


HCMATRIX
fractal_cmatrix_create(int height, int width)
{
    HS_CMATRIX hc = malloc(sizeof(struct S_CMATRIX));

    #ifdef __AVX2__
        #if defined(_WIN64) || defined(_WIN32)
            hc->cmatrix = _aligned_malloc(sizeof(float *) * height, AVX_ALIGNMENT);
            for (int i = 0; i < height; ++i) {
                hc->cmatrix[i] = _aligned_malloc(sizeof(float) * width, AVX_ALIGNMENT);
            } 
        #else // Linux
            hc->cmatrix = aligned_alloc(AVX_ALIGNMENT, sizeof(float *) * height);
            for (int i = 0; i < height; ++i) {
                hc->cmatrix[i] = aligned_alloc(AVX_ALIGNMENT, sizeof(float) * width);
            } 
        #endif
    #else
        hc->cmatrix = malloc(sizeof(float *) * height);
        for (int i = 0; i < height; ++i) {
            hc->cmatrix[i] = malloc(sizeof(float) * width);
        }   
    #endif

    hc->height = height;
    hc->width = width;

    return (HCMATRIX)hc;
}

HCMATRIX
fractal_cmatrix_reshape(HCMATRIX hCmatrix, int height_new, int width_new)
{
    HS_CMATRIX hc = (HS_CMATRIX) hCmatrix;

    #ifdef __AVX2__
        #if defined(_WIN64) || defined(_WIN32)
            for (int i = 0; i < hc->height; ++i) {
                _aligned_free(hc->cmatrix[i]);
            }
            hc->cmatrix = _aligned_realloc(hc->cmatrix, sizeof(float *) * height_new, AVX_ALIGNMENT);
            for (int i = 0; i < height_new; ++i) {
                hc->cmatrix[i] = _aligned_malloc(sizeof(float) * width_new, AVX_ALIGNMENT);
            }
        #else
            // Linux
            for (int i = 0; i < hc->height; ++i) {
                free(hc->cmatrix[i]);
            }
            free(hc->cmatrix);
            hc = fractal_cmatrix_create(height_new, width_new);
        #endif
    #else
        for (int i = 0; i < hc->height; ++i) {
            free(hc->cmatrix[i]);
        }
        hc->cmatrix = realloc(hc->cmatrix, sizeof(float *) * height_new);
        for (int i = 0; i < height_new; ++i) {
            hc->cmatrix[i] = malloc(sizeof(float) * width_new);
        }
    #endif

    hc->height = height_new;
    hc->width = width_new;

    return hc;
}

void
fractal_cmatrix_free(HCMATRIX hCmatrix)
{
    HS_CMATRIX hc = (HS_CMATRIX) hCmatrix;

    #if defined(__AVX2__) && (defined(_WIN32) || defined(_WIN64))
        for (int i = 0; i < hc->height; ++i) {
            _aligned_free(hc->cmatrix[i]);
        }
        _aligned_free(hc->cmatrix);
    #else
        for (int i = 0; i < hc->height; ++i) {
            free(hc->cmatrix[i]);
        }
        free(hc->cmatrix);
    #endif
    free(hc);
}

float *
fractal_cmatrix_value(HCMATRIX hCmatrix, int height, int width)
{
    return &((HS_CMATRIX) hCmatrix)->cmatrix[height][width];
}

float
fractal_cmatrix_max(HCMATRIX hCmatrix)
{
    HS_CMATRIX hc = (HS_CMATRIX) hCmatrix;

    float max_color = 0.0;
    for (int h=0; h<hc->height; ++h) {
        for (int w=0; w<hc->width; ++w) {
            if (hc->cmatrix[h][w] > max_color) {
                max_color = hc->cmatrix[h][w];
            }
        }
    }
    return max_color;
}

void
fractal_cmatrix_save(HCMATRIX hCmatrix, const char * filename, enum FC_Color color)
{
    HS_CMATRIX hc = (HS_CMATRIX) hCmatrix;

    int comp = 3; // rgb

    uint8_t pr, pg, pb;
    uint8_t * data = malloc(hc->height*hc->width*comp);
    for (int r=0; r<hc->height; ++r) {
        for (int c=0; c<hc->width; ++c) {
            fractal_value_to_color(&pr, &pg, &pb, (int)*fractal_cmatrix_value(hCmatrix, r, c), color);
            data[r*(hc->width*3)+(c*3)] = pr;
            data[r*(hc->width*3)+(c*3)+1] = pg;
            data[r*(hc->width*3)+(c*3)+2] = pb;
        }
    }

    stbi_write_png(filename, hc->width, hc->height, comp, data, 0);

    free(data);
}

void
fractal_image_save(uint8_t * image, int width, int height, const char * filename)
{
    stbi_write_png(filename, width, height, 3, image, 0);
}

uint8_t *
fractal_image_create(int height, int width)
{
    return (uint8_t *) malloc(sizeof(uint8_t)*height*width*3);
}

void
fractal_image_free(uint8_t * image)
{
    free(image);
}
