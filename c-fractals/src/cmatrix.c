#include "main.h"


HCMATRIX
fractal_cmatrix_create(int ROWS, int COLS)
{
    HS_CMATRIX hc = malloc(sizeof(struct S_CMATRIX));

    hc->ROWS = ROWS;
    hc->COLS = COLS;
    hc->cmatrix = malloc(sizeof(double *) * ROWS);
	for (int i = 0; i < ROWS; ++i) {
		hc->cmatrix[i] = malloc(sizeof(double) * COLS);
	}
    return (HCMATRIX)hc;
}

HCMATRIX
fractal_cmatrix_reshape(HCMATRIX hCmatrix, int ROWS_new, int COLS_new)
{
    HS_CMATRIX hc = (HS_CMATRIX) hCmatrix;

    for (int i = 0; i < hc->ROWS; ++i) {
		free(hc->cmatrix[i]);
	}

    hc->ROWS = ROWS_new;
    hc->COLS = COLS_new;
    
    hc->cmatrix = realloc(hc->cmatrix, sizeof(double *) * ROWS_new);
	for (int i = 0; i < ROWS_new; ++i) {
		hc->cmatrix[i] = malloc(sizeof(double) * COLS_new);
	}
    return hc;
}

void
fractal_cmatrix_free(HCMATRIX hCmatrix)
{
    HS_CMATRIX hc = (HS_CMATRIX) hCmatrix;

    for (int i = 0; i < hc->ROWS; ++i) {
		free(hc->cmatrix[i]);
	}
    free(hc->cmatrix);
    free(hc);
}

double *
fractal_cmatrix_value(HCMATRIX hCmatrix, int row, int col)
{
    return &((HS_CMATRIX) hCmatrix)->cmatrix[row][col];
}
