#include "fractal_color.h"
#include "fractal_cuda.h"
#include "raylib.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <complex.h>


#define NUM_THREADS 12
#define VECSIZE 8

#ifdef CUDA
#define WIDTH 1920 // (VECSIZE*100)
#define HEIGHT 1080 // WIDTH
#else
#define WIDTH 1280 // (VECSIZE*100)
#define HEIGHT 720 // WIDTH
#endif

#define LYAPUNOV_SEQUENCE "ABAABBAA"


struct FractalProperties fp;
enum FC_Color f_color;
float animation_speed;
bool animate;
bool update;
bool show_info;
bool show_position;
float aspect_ratio;
int counter;


void
reset(bool view_only)
{
    aspect_ratio = (float)WIDTH/HEIGHT;
    animation_speed = 1;

    fp.x_start=-2;
    fp.x_end=2;
    fp.y_start=-2/aspect_ratio;
    fp.y_end=2/aspect_ratio;
    fp.width=WIDTH;
    fp.height=HEIGHT;
    

    if (view_only) return;

    fp.frac=FC_FRAC_Z2;
    fp.mode=FC_MODE_JULIA;
    fp.c_real=1;
    fp.c_imag=1;
    fp.R=2;
    fp.max_iterations=250;
    fp.sequence=LYAPUNOV_SEQUENCE;
    fp.sequence_length=sizeof(LYAPUNOV_SEQUENCE)-1;

    f_color = FC_COLOR_ULTRA;

    animate = true;
    update = true;
    show_info = true;
    show_position = false;
}


void
handle_user_input()
{
    if (animate) {
        if (IsKeyPressed(KEY_LEFT_CONTROL)){
            animation_speed -= 0.1;
            if (animation_speed < 0.1) animation_speed = 0.1;
        }
        if (IsKeyPressed(KEY_RIGHT_CONTROL)){
            animation_speed += 0.1;
        }
        update = true;
    }

    bool shift_pressed = IsKeyDown(KEY_LEFT_SHIFT);

    if (GetMouseWheelMove() == 1) {
        if ((fp.x_end-fp.x_start > 0.01) && (fp.y_end-fp.y_start>0.01)) {
            fp.x_start += shift_pressed ?  0.01 : 0.05;
            fp.y_start += (shift_pressed ?  0.01 : 0.05)/aspect_ratio;
            fp.x_end -= shift_pressed ?  0.01 : 0.05;
            fp.y_end -= (shift_pressed ?  0.01 : 0.05)/aspect_ratio;
            update = true;
        }
    }
    if (GetMouseWheelMove() == -1) {
        fp.x_start -= shift_pressed ?  0.01 : 0.05;
        fp.y_start -= (shift_pressed ?  0.01 : 0.05)/aspect_ratio;
        fp.x_end += shift_pressed ?  0.01 : 0.05;
        fp.y_end += (shift_pressed ?  0.01 : 0.05)/aspect_ratio;
        update = true;
    }

    /* Pan around with WASD keys */
    if (IsKeyDown(KEY_S)){
        fp.y_start += shift_pressed ?  0.001 : 0.01;
        fp.y_end += shift_pressed ?  0.001 : 0.01;
        update = true;
    }
    if (IsKeyDown(KEY_W)){
        fp.y_start -= shift_pressed ?  0.001 : 0.01;
        fp.y_end -= shift_pressed ?  0.001 : 0.01;
        update = true;
    }
    if (IsKeyDown(KEY_A)){
        fp.x_start -= shift_pressed ?  0.001 : 0.01;
        fp.x_end -= shift_pressed ?  0.001 : 0.01;
        update = true;
    }
    if (IsKeyDown(KEY_D)){
        fp.x_start += shift_pressed ?  0.001 : 0.01;
        fp.x_end += shift_pressed ?  0.001 : 0.01;
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
    /* Toggle position */
    if (IsKeyPressed(KEY_F2)){
        show_position = !show_position;
    }
    /* Toggle Fullscreen */
    if (IsKeyPressed(KEY_F11)){
        ToggleFullscreen();
    }
}


int
main(void)
{
    reset(false);

    #ifdef CUDA
        int * cuda_image = fractal_image_create(HEIGHT, WIDTH);
        fractal_cuda_init(WIDTH, HEIGHT);
    #else
        HCMATRIX hc = fractal_cmatrix_create(HEIGHT, WIDTH);
    #endif

    Color * image;
	image = malloc(WIDTH*HEIGHT*sizeof(Color));
	
	Image img;
	img.data = image;
	img.width = WIDTH;
	img.height = HEIGHT;
	img.mipmaps = 1;
	img.format = UNCOMPRESSED_R8G8B8A8;

    InitWindow(WIDTH, HEIGHT, "raylib fractals"); 

    SetTargetFPS(GetMonitorRefreshRate(0));
    
    Texture2D texture = LoadTextureFromImage(img);

    bool firstrun = true;
    int fps = 0;
    float frametime = 0;
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

        if (fp.mode == FC_MODE_MANDELBROT || fp.mode == FC_MODE_LYAPUNOV) animate = false;

        if (animate) {
            fp.c_real = 0.7885 * cosf(counter / (2 * PI) / 20 * animation_speed);
            fp.c_imag = 0.7885 * sinf(counter / (2 * PI) / 10 * animation_speed);
            counter++;
            update = true;
        }

        if (update) {
            // Do the actual fractal computation
            #ifdef CUDA
                fractal_cuda_get_colors(cuda_image, &fp);
            #else
                fractal_avxf_get_colors_th(hc, &fp, NUM_THREADS);
            #endif

            // Convert the values to a color image
            for (int row=0; row<HEIGHT; ++row) {
                for (int col=0; col<WIDTH; ++col) {
                    float r, g, b;
                    #ifdef CUDA
                        fractal_value_to_color(&r, &g, &b, cuda_image[row*WIDTH+col], f_color);
                    #else
                        fractal_value_to_color(&r, &g, &b, (int)*fractal_cmatrix_value(hc, row, col), f_color);
                    #endif
                    Color c;
                    c.a = 255;
                    c.r = r*255;
                    c.g = g*255;
                    c.b = b*255;
                    image[row*WIDTH+col] = c;
                }
            }
        }

        // Update fps and frametime less frequently to improve readability
        if (counter % 10 == 0) {
            fps = GetFPS();
            frametime = GetFrameTime()*1000;
        } else {
            GetFPS(); // otherwise values will take more time to converge
        }

		BeginDrawing();

		if (update) UpdateTexture(texture, image);
		DrawTexture(texture, 0, 0, WHITE);
		
        if (show_info) {
            sprintf(buf, "FPS: %d\n"
                         "Frametime: %3.1f ms\n"
                         "Animation speed: %.1f",
                         fps, frametime, animation_speed);
            DrawText(buf, 10, 5, 19, RED);

            sprintf(buf, "\n\n\n"
                         "WASD: Pan around\n"
                         "Space: Pause\n"
                         "1: Colors\n"
                         "2: Fractals\n"
                         "3: Modes\n"
                         "R: Reset\n"
                         "F2: Toggle position\n"
                         "F1: Toggle OSD\n");
            DrawText(buf, 10, 5, 20, PURPLE);

            if (fp.mode == FC_MODE_JULIA) {
                sprintf(buf, "c = %5.02f + %5.02f j", fp.c_real,fp.c_imag);
                DrawText(buf, WIDTH-200, 5, 20, GREEN);
            }
        }

        if (show_position){
            sprintf(buf, "%.4f, %.4f",
                    fp.x_start + GetMouseX()/fp.width*(fp.x_end-fp.x_start),
                    fp.y_start + GetMouseY()/fp.height*(fp.y_end-fp.y_start));
            DrawText(buf, 10, HEIGHT-30, 20, PINK);

            DrawLine(0, GetMouseY(), WIDTH, GetMouseY(), PINK);
            DrawLine(GetMouseX(), 0, GetMouseX(), HEIGHT, PINK);
        }

        EndDrawing();
    }

    CloseWindow();        // Close window and OpenGL context

    #ifdef CUDA
        fractal_image_free(cuda_image);
        fractal_cuda_clean();
    #else
        fractal_cmatrix_free(hc);
    #endif
    
    return 0;
}
