#include "main.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"


HCMATRIX
fractal_cmatrix_create(int ROWS, int COLS)
{
    HS_CMATRIX hc = malloc(sizeof(struct S_CMATRIX));

    #ifdef __AVX2__
        #if defined(_WIN64) || defined(_WIN32)
            hc->cmatrix = _aligned_malloc(sizeof(FRACDTYPE *) * ROWS, AVX_ALIGNMENT);
            for (int i = 0; i < ROWS; ++i) {
                hc->cmatrix[i] = _aligned_malloc(sizeof(FRACDTYPE) * COLS, AVX_ALIGNMENT);
            } 
        #else // Linux
            hc->cmatrix = aligned_alloc(AVX_ALIGNMENT, sizeof(FRACDTYPE *) * ROWS);
            for (int i = 0; i < ROWS; ++i) {
                hc->cmatrix[i] = aligned_alloc(AVX_ALIGNMENT, sizeof(FRACDTYPE) * COLS);
            } 
        #endif
    #else
        hc->cmatrix = malloc(sizeof(FRACDTYPE *) * ROWS);
        for (int i = 0; i < ROWS; ++i) {
            hc->cmatrix[i] = malloc(sizeof(FRACDTYPE) * COLS);
        }   
    #endif

    hc->ROWS = ROWS;
    hc->COLS = COLS;

    return (HCMATRIX)hc;
}

HCMATRIX
fractal_cmatrix_reshape(HCMATRIX hCmatrix, int ROWS_new, int COLS_new)
{
    HS_CMATRIX hc = (HS_CMATRIX) hCmatrix;

    #ifdef __AVX2__
        #if defined(_WIN64) || defined(_WIN32)
            for (int i = 0; i < hc->ROWS; ++i) {
                _aligned_free(hc->cmatrix[i]);
            }
            hc->cmatrix = _aligned_realloc(hc->cmatrix, sizeof(FRACDTYPE *) * ROWS_new, AVX_ALIGNMENT);
            for (int i = 0; i < ROWS_new; ++i) {
                hc->cmatrix[i] = _aligned_malloc(sizeof(FRACDTYPE) * COLS_new, AVX_ALIGNMENT);
            }
        #else
            // Linux
            for (int i = 0; i < hc->ROWS; ++i) {
                free(hc->cmatrix[i]);
            }
            free(hc->cmatrix);
            hc = fractal_cmatrix_create(ROWS_new, COLS_new);
        #endif
    #else
        for (int i = 0; i < hc->ROWS; ++i) {
            free(hc->cmatrix[i]);
        }
        hc->cmatrix = realloc(hc->cmatrix, sizeof(FRACDTYPE *) * ROWS_new);
        for (int i = 0; i < ROWS_new; ++i) {
            hc->cmatrix[i] = malloc(sizeof(FRACDTYPE) * COLS_new);
        }
    #endif

    hc->ROWS = ROWS_new;
    hc->COLS = COLS_new;

    return hc;
}

void
fractal_cmatrix_free(HCMATRIX hCmatrix)
{
    HS_CMATRIX hc = (HS_CMATRIX) hCmatrix;

    #if defined(__AVX2__) && (defined(_WIN32) || defined(_WIN64))
        for (int i = 0; i < hc->ROWS; ++i) {
            _aligned_free(hc->cmatrix[i]);
        }
        _aligned_free(hc->cmatrix);
    #else
        for (int i = 0; i < hc->ROWS; ++i) {
            free(hc->cmatrix[i]);
        }
        free(hc->cmatrix);
    #endif
    free(hc);
}

FRACDTYPE *
fractal_cmatrix_value(HCMATRIX hCmatrix, int row, int col)
{
    return &((HS_CMATRIX) hCmatrix)->cmatrix[row][col];
}

FRACDTYPE
fractal_cmatrix_max(HCMATRIX hCmatrix)
{
    HS_CMATRIX hc = (HS_CMATRIX) hCmatrix;

    FRACDTYPE max_color = 0.0;
    for (int row=0; row<hc->ROWS; ++row) {
        for (int col=0; col<hc->COLS; ++col) {
            if (hc->cmatrix[row][col] > max_color) {
                max_color = hc->cmatrix[row][col];
            }
        }
    }
    return max_color;
}

void
fractal_cmatrix_save(HCMATRIX hCmatrix, const char * filename)
{
    HS_CMATRIX hc = (HS_CMATRIX) hCmatrix;

    int comp = 3; // rgb

    float pr, pg, pb;
    char * data = malloc(hc->ROWS*hc->COLS*comp);
    for (int r=0; r<hc->ROWS; ++r) {
        for (int c=0; c<hc->COLS; ++c) {
            value_to_rgb_ultra(&pr, &pg, &pb, (int)*fractal_cmatrix_value(hCmatrix, r, c));
            data[r*(hc->COLS*3)+(c*3)] = (char) (pr*255);
            data[r*(hc->COLS*3)+(c*3)+1] = (char) (pg*255);
            data[r*(hc->COLS*3)+(c*3)+2] = (char) (pb*255);
        }
    }

    stbi_write_png(filename, hc->COLS, hc->ROWS, comp, data, 0);

    free(data);
}
