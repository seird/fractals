#include "flames.h"

#include "../stb_image_write.h"
#include "variations.h"
#include <math.h>
#include <stddef.h>


struct Histogram {
    size_t * data;
    int width;
    int height;
    size_t max;
};


struct Colors {
    struct RGB * data;
    int width;
    int height;
};


static void
flames_update(struct Histogram * hist, struct Colors * colors, struct Vector2 point, struct RGB color)
{
    float low = -1.0f; // some symmetric range we are interested in
    float high = -low;
    int x = (point.x - low) / (high-low) * (hist->width-1); // scale the point from [low, high] to [0, width-1]
    int y = (point.y - low) / (high-low) * (hist->height-1); // scale the point from [low, high] to [0, height-1]

    if (x >= 0 && x < hist->width && y >= 0 && y < hist->height) {
        if ((hist->data[x+y*hist->width] += 1) > hist->max) {
            hist->max = hist->data[x+y*hist->width];
        }
        colors->data[x+y*hist->width] = scale_color(
            add_colors(colors->data[x+y*hist->width], color),
            0.5f
        );
    }
}


static void
flames_chaos_game(struct Histogram * hist, struct Colors * colors, int N)
{
    struct Vector2 point = random_point(-1.0f, 1.0f);
    struct RGB c = random_color();
    int num_variations = flames_num_variations();
    hist->max = 0;
    
    for (int i=0; i<N; ++i) {
        int v = rand() % num_variations;
        variationfunc_t variationfunc = flames_variationfunc_get(v);

        point = (*variationfunc)(point);

        c = scale_color(
            add_colors(c, flames_variationcolor_get(v)),
            0.5f
        );

        // F_final
        point = variation_sinusoidal(point);
        c = scale_color(
            add_colors(c, (struct RGB){1, 1, 1}),
            0.5f
        );

        // update hist and colors
        if (i >= 20) {
            flames_update(hist, colors, point, c);
        }
    }
}


static void
flames_render(uint8_t * image, struct Histogram * hist, struct Colors * colors, int supersample, float gamma)
{
    for (int h=0; h<hist->height; h+=supersample) {
        for (int w=0; w<hist->width; w+=supersample) {
            // reduce supersample
            float avg_freq = 0.0f;
            struct RGB avg_color = {0, 0, 0};
            for (int h_=h; h_<h+supersample; ++h_) {
                for (int w_=w; w_<w+supersample; ++w_) {
                    avg_freq += hist->data[w_+h_*hist->height];
                    avg_color = add_colors(avg_color, colors->data[w_+h_*hist->height]);
                }
            }
            avg_freq /= supersample*supersample;
            avg_color = scale_color(avg_color, 1.0f/(supersample*supersample));

            float alpha = logf(avg_freq)/logf(hist->max); // hist->max should be replaced with max of averaged hist

            uint8_t * pixel = image+3*(w/supersample)+3*(h/supersample)*(hist->height/supersample);

            *pixel     = avg_color.r*powf(alpha, 1/gamma)*255;
            *(pixel+1) = avg_color.g*powf(alpha, 1/gamma)*255;
            *(pixel+2) = avg_color.b*powf(alpha, 1/gamma)*255;
        }
    }
}


void
flames_get_colors(struct FractalProperties * fp)
{
    int supersample = fp->flame.supersample < 1 ? 1 : fp->flame.supersample;
    float gamma = fp->flame.gamma <= 0 ? 1.0f : fp->flame.gamma;

    struct Histogram hist = {
        .data = (size_t *) calloc((fp->width*supersample)*(fp->height*supersample), sizeof(size_t)),
        .width = fp->width*supersample,
        .height = fp->height*supersample
    };

    struct Colors colors = {
        .data = (struct RGB *) calloc((fp->width*supersample)*(fp->height*supersample), sizeof(struct RGB)),
        .width = fp->width*supersample,
        .height = fp->height*supersample
    };

    uint8_t * image = (uint8_t *) calloc(3*fp->width*fp->height, sizeof(uint8_t));

    for (int i=0; i<supersample*fp->flame.num_chaos_games; ++i) {
        flames_chaos_game(&hist, &colors, fp->flame.chaos_game_length);
    }

    flames_render(image, &hist, &colors, supersample, gamma);

    stbi_write_png(fp->flame.savename, fp->width, fp->height, 3, image, 0);

    free(hist.data);
    free(colors.data);
    free(image);
}
