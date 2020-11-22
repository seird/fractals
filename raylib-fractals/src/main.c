#include "fractal_color.h"
#include "raylib.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <complex.h>


#define NUM_THREADS 10
#define WIDTH (NUM_THREADS*8*8)
#define HEIGHT (NUM_THREADS*8*8)


int
main(void)
{
    HCMATRIX hc = fractal_cmatrix_create(WIDTH, HEIGHT);

    struct FractalProperties fp = {
        .x_start=-1.5,
        .x_end=1.5,
        .y_start=-1.5,
        .y_end=1.5,
        .width=WIDTH,
        .height=HEIGHT,
        .frac=FC_FRAC_Z2,
        .mode=FC_MODE_JULIA,
        .c_real=1,
        .c_imag=1,
        .R=2,
        .max_iterations=250,
    };

    Color * image;
	image = malloc(WIDTH*HEIGHT*sizeof(Color));
	
	Image img;
	img.data = image;
	img.width = WIDTH;
	img.height = HEIGHT;
	img.mipmaps = 1;
	img.format = UNCOMPRESSED_R8G8B8A8;

    InitWindow(WIDTH, HEIGHT, "raylib fractals"); 

    SetTargetFPS(100);
    
    Texture2D texture = LoadTextureFromImage(img);

    bool update = false;
    bool animate = true;
    bool firstrun = true;
    bool show_info = true;
    int counter = 0;
    while (!WindowShouldClose())    // Detect window close button or ESC key
    {
		char buf[1024];

        if (firstrun) {
			update = true;
			firstrun = false;
		} else {
			update = false;
		}

        if (animate) {
            fp.c_real = 0.7885 * cosf(counter / (2 * PI) / 20);
            fp.c_imag = 0.7885 * sinf(counter / (2 * PI) / 10);
            counter++;
            update = true;
        }

        if (GetMouseWheelMove() == 1) {
            if ((fp.x_end-fp.x_start > 0.1) && (fp.y_end-fp.y_start>0.1)) {
			    fp.x_start += 0.05;
                fp.y_start += 0.05;
                fp.x_end -= 0.05;
                fp.y_end -= 0.05;
                update = true;
            }
		}
		if (GetMouseWheelMove() == -1) {
            fp.x_start -= 0.05;
			fp.y_start -= 0.05;
			fp.x_end += 0.05;
			fp.y_end += 0.05;
			update = true;
		}

        /* Pan around with WASD keys */
        if (IsKeyDown(KEY_S)){
            fp.y_start += 0.01;
			fp.y_end += 0.01;
			update = true;
        }
        if (IsKeyDown(KEY_W)){
            fp.y_start -= 0.01;
			fp.y_end -= 0.01;
			update = true;
        }
        if (IsKeyDown(KEY_A)){
            fp.x_start -= 0.01;
			fp.x_end -= 0.01;
			update = true;
        }
        if (IsKeyDown(KEY_D)){
            fp.x_start += 0.01;
			fp.x_end += 0.01;
			update = true;
        }
        /* Pause the animation */
        if (IsKeyPressed(KEY_SPACE)){
            animate = !animate;
        }
        /* Toggle osd */
        if (IsKeyPressed(KEY_F1)){
            show_info = !show_info;
        }

        if (update) {
            // fractal_avxf_get_colors_th(hc, &fp, NUM_THREADS);
            // fractal_get_colors_th(hc, &fp, NUM_THREADS);
            fractal_avxf_get_colors(hc, &fp);

            for (int row=0; row<HEIGHT; ++row) {
                for (int col=0; col<WIDTH; ++col) {
                    float r, g, b;
                    fractal_value_to_color(&r, &g, &b, (int)*fractal_cmatrix_value(hc, row, col), FC_COLOR_ULTRA);
                    Color c;
                    c.a = 255;
                    c.r = r*255;
                    c.g = g*255;
                    c.b = b*255;
                    image[row*WIDTH+col] = c;
                }
            }
        }

		BeginDrawing();
		if ( update ) UpdateTexture(texture, image);
		DrawTexture(texture, 0, 0, WHITE);
		
        if (show_info) {
            sprintf(buf, "FPS: %d", GetFPS());
            DrawText(buf, 0, 0, 18, PURPLE);
            sprintf(buf, "\nWASD: Pan around\nSpace: Pause\nF1: Toggle OSD");
            DrawText(buf, 0, 0, 18, PURPLE);
        }
		
        EndDrawing();
    }

    CloseWindow();        // Close window and OpenGL context

    fractal_cmatrix_free(hc);
    
    return 0;
}
