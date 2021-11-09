### C Functions

- HCMATRIX fractal_cmatrix_create(int height, int width);
- HCMATRIX fractal_cmatrix_reshape(HCMATRIX hCmatrix, int height_new, int width_new);
- void fractal_cmatrix_free(HCMATRIX hCmatrix);
- float * fractal_cmatrix_value(HCMATRIX hCmatrix, int height, int width);
- void fractal_get_colors(HCMATRIX hCmatrix, struct FractalProperties * fp);
- void fractal_get_colors_th(HCMATRIX hCmatrix, struct FractalProperties * fp, int num_threads);
- void fractal_avxf_get_colors(HCMATRIX hCmatrix, struct FractalProperties * fp);
- void fractal_avxf_get_colors_th(HCMATRIX hCmatrix, struct FractalProperties * fp, int num_threads);
- float fractal_cmatrix_max(HCMATRIX hCmatrix);
- void fractal_value_to_color(uint8_t * r, uint8_t * g, uint8_t * b, int value, enum FC_Color color);
- colorfunc_t fractal_colorfunc_get(enum FC_Color color);
- void fractal_cmatrix_save(HCMATRIX hCmatrix, const char * filename, enum FC_Color color);


### Cuda Functions

- bool fractal_cuda_init(int width, int height);
- void fractal_cuda_clean();
- uint8_t * fractal_image_create(int height, int width);
- void fractal_image_free(uint8_t * image);
- void fractal_cuda_get_colors(uint8_t * image, struct FractalProperties * fp);
- void fractal_image_save(uint8_t * image, int width, int height, const char * filename);
