#include "fractal_color.h"
#include "raylib.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <complex.h>


#define NUM_THREADS 12
#define VECSIZE 8
#define WIDTH (VECSIZE*130)
#define HEIGHT WIDTH


struct FractalProperties fp;
enum FC_Color f_color;
float animation_speed;
bool animate;
bool update;
bool show_info;
int counter;


void
reset(bool view_only)
{
    fp.x_start=-2;
    fp.x_end=2;
    fp.y_start=-2;
    fp.y_end=2;
    fp.width=WIDTH;
    fp.height=HEIGHT;
    animation_speed = 1;

    if (view_only) return;

    fp.frac=FC_FRAC_Z2;
    fp.mode=FC_MODE_JULIA;
    fp.c_real=1;
    fp.c_imag=1;
    fp.R=2;
    fp.max_iterations=250;

    f_color = FC_COLOR_ULTRA;

    animate = true;
    update = true;
    show_info = true;
}


void
handle_user_input()
{
    if (fp.mode == FC_MODE_MANDELBROT) animate = false;

    if (animate) {
        if (IsKeyPressed(KEY_LEFT_CONTROL)){
            animation_speed -= 0.1;
            if (animation_speed < 0.1) animation_speed = 0.1;
        }
        if (IsKeyPressed(KEY_RIGHT_CONTROL)){
            animation_speed += 0.1;
        }
        
        fp.c_real = 0.7885 * cosf(counter / (2 * PI) / 20 * animation_speed);
        fp.c_imag = 0.7885 * sinf(counter / (2 * PI) / 10 * animation_speed);
        counter++;
        update = true;
    }

    bool shift_pressed = IsKeyDown(KEY_LEFT_SHIFT);

    if (GetMouseWheelMove() == 1) {
        if ((fp.x_end-fp.x_start > 0.01) && (fp.y_end-fp.y_start>0.01)) {
            fp.x_start += 0.05 * !shift_pressed + 0.01 * shift_pressed;
            fp.y_start += 0.05 * !shift_pressed + 0.01 * shift_pressed;
            fp.x_end -= 0.05 * !shift_pressed + 0.01 * shift_pressed;
            fp.y_end -= 0.05 * !shift_pressed + 0.01 * shift_pressed;
            update = true;
        }
    }
    if (GetMouseWheelMove() == -1) {
        fp.x_start -= 0.05 * !shift_pressed + 0.01 * shift_pressed;
        fp.y_start -= 0.05 * !shift_pressed + 0.01 * shift_pressed;
        fp.x_end += 0.05 * !shift_pressed + 0.01 * shift_pressed;
        fp.y_end += 0.05 * !shift_pressed + 0.01 * shift_pressed;
        update = true;
    }

    /* Pan around with WASD keys */
    if (IsKeyDown(KEY_S)){
        fp.y_start += 0.01 * !shift_pressed + 0.001 * shift_pressed;
        fp.y_end += 0.01 * !shift_pressed + 0.001 * shift_pressed;
        update = true;
    }
    if (IsKeyDown(KEY_W)){
        fp.y_start -= 0.01 * !shift_pressed + 0.001 * shift_pressed;
        fp.y_end -= 0.01 * !shift_pressed + 0.001 * shift_pressed;
        update = true;
    }
    if (IsKeyDown(KEY_A)){
        fp.x_start -= 0.01 * !shift_pressed + 0.001 * shift_pressed;
        fp.x_end -= 0.01 * !shift_pressed + 0.001 * shift_pressed;
        update = true;
    }
    if (IsKeyDown(KEY_D)){
        fp.x_start += 0.01 * !shift_pressed + 0.001 * shift_pressed;
        fp.x_end += 0.01 * !shift_pressed + 0.001 * shift_pressed;
        update = true;
    }
    /* Cycle colors */
    if (IsKeyPressed(KEY_ONE)){
        f_color = (f_color + 1) % FC_COLOR_NUM_ENTRIES;
        update = true;
    }
    /* Cycle fractals */
    if (IsKeyPressed(KEY_TWO)){
        fp.frac = (fp.frac + 1) % FC_FRAC_NUM_ENTRIES;
        update = true;
    }
    /* Cycle modes */
    if (IsKeyPressed(KEY_THREE)){
        fp.mode = (fp.mode + 1) % FC_MODE_NUM_ENTRIES;
        update = true;
    }
    /* Pause the animation */
    if (IsKeyPressed(KEY_SPACE)){
        animate = !animate;
    }
    /* Reset */
    if (IsKeyPressed(KEY_R)){
        reset(true);
    }
    /* Toggle osd */
    if (IsKeyPressed(KEY_F1)){
        show_info = !show_info;
    }
}


int
main(void)
{
    reset(false);

    HCMATRIX hc = fractal_cmatrix_create(WIDTH, HEIGHT);

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

    bool firstrun = true;
    counter = 0;
    while (!WindowShouldClose())    // Detect window close button or ESC key
    {
		char buf[1024];

        if (firstrun) {
			update = true;
			firstrun = false;
		} else {
			update = false;
		}

        handle_user_input();

        if (update) {
            // Do the actual fractal computation
            fractal_avxf_get_colors_th(hc, &fp, NUM_THREADS);

            // Convert the values to a color image
            for (int row=0; row<HEIGHT; ++row) {
                for (int col=0; col<WIDTH; ++col) {
                    float r, g, b;
                    fractal_value_to_color(&r, &g, &b, (int)*fractal_cmatrix_value(hc, row, col), f_color);
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
            sprintf(buf, "FPS: %d\nAnimation speed: %.1f", GetFPS(), animation_speed);
            DrawText(buf, 10, 5, 19, RED);

            sprintf(buf, "\n\n"
                         "WASD: Pan around\n"
                         "Space: Pause\n"
                         "1: Colors\n"
                         "2: Fractals\n"
                         "3: Modes\n"
                         "R: Reset\n"
                         "F1: Toggle OSD\n");
            DrawText(buf, 10, 5, 20, PURPLE);

            if (fp.mode == FC_MODE_JULIA) {
                sprintf(buf, "c = %5.02f + %5.02f j", fp.c_real,fp.c_imag);
                DrawText(buf, WIDTH-200, 0, 20, GREEN);
            }
        }
        EndDrawing();
    }

    CloseWindow();        // Close window and OpenGL context

    fractal_cmatrix_free(hc);
    
    return 0;
}
