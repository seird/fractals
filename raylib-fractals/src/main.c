// Include the relevant fractal api header
#ifdef CUDA
#include "../../cuda-fractals/include/fractal_cuda.h"
#elif defined(OPENCL)
#include "../../opencl-fractals/include/fractal_opencl.h"
#else
#include "../../c-fractals/include/fractal_color.h"
#endif

#include "animations.h"
#include "ui_strings.h"
#include "raylib.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <complex.h>


#define NUM_THREADS 24
#define MAX_ITERATIONS 250
#define R_DEFAULT 2

#if defined(CUDA) || defined(OPENCL)
#define WIDTH 2560
#define HEIGHT 1440
#else
#define WIDTH 1920 // multiple of (vector size) --> AVX2: 8, AVX512: 16
#define HEIGHT 1080
#endif

#define LYAPUNOV_SEQUENCE "ABAAB"


struct FractalProperties fp;
enum FC_Color f_color;
#if !(defined(CUDA) || defined(OPENCL))
colorfunc_t colorfunc;
#endif
float animation_speed;
bool animate;
bool update;
bool show_info;
bool show_position;
float aspect_ratio;
int step;
animationfunc_t animationfunc;
enum Animation animation;


void
reset(bool view_only)
{
    aspect_ratio = (float)WIDTH/HEIGHT;
    animation_speed = 1;

    fp.x_start=-2;
    fp.x_end=2;
    fp.y_start=-2/aspect_ratio;
    fp.y_end=2/aspect_ratio;
    

    if (view_only) return;

    fp.frac=FC_FRAC_Z2;
    fp.mode=FC_MODE_JULIA;
    fp.c_real=1;
    fp.c_imag=1;
    fp.R=R_DEFAULT;
    fp.max_iterations=MAX_ITERATIONS;
    fp.sequence=LYAPUNOV_SEQUENCE;
    fp.sequence_length=sizeof(LYAPUNOV_SEQUENCE)-1;

    f_color = FC_COLOR_ULTRA;
    fp.color = f_color;
    #if !(defined(CUDA) || defined(OPENCL))
    colorfunc = fractal_colorfunc_get(f_color);
    #endif

    animate = true;
    update = true;
    show_info = true;
    show_position = false;
    animationfunc = animationfunc_get(ANIMATION_DEFAULT);
    animation = ANIMATION_DEFAULT;
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
        #if !(defined(CUDA) || defined(OPENCL))
        colorfunc = fractal_colorfunc_get(f_color);
        #endif
        fp.color = f_color;
        update = true;
    }
    /* Cycle fractals */
    if (IsKeyPressed(KEY_TWO)){
        fp.frac = (fp.frac + 1) % FC_FRAC_NUM_ENTRIES;
        update = true;
    }
    /* Cycle modes */
    if (IsKeyPressed(KEY_THREE)){
        fp.mode = (fp.mode + 1) % (FC_MODE_NUM_ENTRIES);
        update = true;
    }
    /* Cycle animations */
    if (IsKeyPressed(KEY_FOUR)){
        animation = (animation + 1) % NUM_ANIMATIONS;
        animationfunc = animationfunc_get(animation);
        fp.max_iterations = MAX_ITERATIONS;
        fp.R = R_DEFAULT;
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
    /* Manually move the animation forward */
    if (IsKeyDown(KEY_RIGHT)) {
        update = true;
        animate = false;
        step++;
        animationfunc(&fp, step, animation_speed);
    }
    /* Manually move the animation backward */
    if (IsKeyDown(KEY_LEFT)) {
        update = true;
        animate = false;
        step--;
        animationfunc(&fp, step, animation_speed);
    }
}


int
main(void)
{
    reset(false);

    #ifdef CUDA
        uint8_t * cuda_image = fractal_cuda_image_create(HEIGHT, WIDTH);
        fractal_cuda_init(WIDTH, HEIGHT);
    #elif defined(OPENCL)
        uint8_t * opencl_image = fractal_opencl_image_create(HEIGHT, WIDTH);
        fractal_opencl_init(WIDTH, HEIGHT);
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

        if (fp.mode == FC_MODE_LYAPUNOV) animate = false;
        if (fp.mode == FC_MODE_MANDELBROT && (animation == ANIMATION_DEFAULT || animation == ANIMATION_DEFAULT_ITERATIONS)) animate = false;

        if (animate) {
            animationfunc(&fp, step, animation_speed);
            step++;
            update = true;
        }

        if (update) {
            // Do the actual fractal computation
            #ifdef CUDA
                fractal_cuda_get_colors(cuda_image, &fp);
            #elif defined(OPENCL)
                fractal_opencl_get_colors(opencl_image, &fp);
            #else
                fractal_avxf_get_colors_th(hc, &fp, NUM_THREADS);
                // fractal_avx512f_get_colors_th(hc, &fp, NUM_THREADS);
            #endif

            // Convert the values to a color image
            for (int h=0; h<HEIGHT; ++h) {
                for (int w=0; w<WIDTH; ++w) {
                    Color c;
                    c.a = 255;
                    #ifdef CUDA
                        c.r = cuda_image[h*WIDTH*3 + w*3];
                        c.g = cuda_image[h*WIDTH*3 + w*3 + 1];
                        c.b = cuda_image[h*WIDTH*3 + w*3 + 2];
                    #elif defined(OPENCL)
                        c.r = opencl_image[h*WIDTH*3 + w*3];
                        c.g = opencl_image[h*WIDTH*3 + w*3 + 1];
                        c.b = opencl_image[h*WIDTH*3 + w*3 + 2];
                    #else
                        uint8_t r, g, b;
                        colorfunc(&r, &g, &b, (int)*fractal_cmatrix_value(hc, h, w));
                        c.r = r;
                        c.g = g;
                        c.b = b;
                    #endif
                    image[h*WIDTH+w] = c;
                }
            }
        }

        // Update fps and frametime less frequently to improve readability
        if (step % 10 == 0) {
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
                         "Space: %s\n"
                         "1: Color: %s\n"
                         "2: Fractal: %s\n"
                         "3: Mode: %s\n"
                         "4: Animation: %s\n"
                         "R: Reset\n"
                         "F2: Toggle position\n"
                         "F1: Toggle OSD\n",
                         (animate ? "Pause" : "Unpause"), color_str[fp.color], fractal_str[fp.frac], mode_str[fp.mode], animation_str[animation]);
            DrawText(buf, 10, 5, 20, PURPLE);

            if (fp.mode == FC_MODE_JULIA || fp.mode == FC_MODE_NEWTON) {
                sprintf(buf, "c = %5.02f + %5.02f j", fp.c_real,fp.c_imag);
                DrawText(buf, WIDTH-200, 65, 20, GREEN);
            }

            sprintf(buf, "ITERS = %d\nR = %3.02f", fp.max_iterations, fp.R);
            DrawText(buf, WIDTH-200, 5, 20, GOLD);
        }

        if (show_position){
            sprintf(buf, "%.4f, %.4f",
                    fp.x_start + (float)GetMouseX()/WIDTH*(fp.x_end-fp.x_start),
                    fp.y_start + (float)GetMouseY()/HEIGHT*(fp.y_end-fp.y_start));
            DrawText(buf, 10, HEIGHT-30, 20, PINK);

            DrawLine(0, GetMouseY(), WIDTH, GetMouseY(), PINK);
            DrawLine(GetMouseX(), 0, GetMouseX(), HEIGHT, PINK);
        }

        EndDrawing();
    }

    CloseWindow();        // Close window and OpenGL context

    #ifdef CUDA
        fractal_cuda_image_free(cuda_image);
        fractal_cuda_clean();
    #elif defined(OPENCL)
        fractal_opencl_image_free(opencl_image);
        fractal_opencl_clean();
    #else
        fractal_cmatrix_free(hc);
    #endif
    
    return 0;
}
