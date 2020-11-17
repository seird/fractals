#include <windows.h>

#include <stdint.h>
#include <stdbool.h>
#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <math.h>

#include <GLFW\glfw3.h>
#include "fractal_color.h"


#define PI 3.14159
#define R_escape 2.0

#define HEIGHT 800
#define WIDTH 800
#define MAX_ITERATIONS 250

#define PAN_STEP 15

#define ZOOMLIMIT 1e-5 // 1e-12

float x_start, x_end, x_step, y_start, y_end, y_step;

bool update;

enum ColorFunction {
	CF_DEFAULT = 1,
	CF_THREADED,
	CF_AVX,
	CF_AVX_THREADED
};

void
view_reset()
{
	x_start = -R_escape;
	x_end = R_escape;

	y_start = -R_escape;
	y_end = R_escape;

	x_step = 2 * R_escape / WIDTH;
	y_step = 2 * R_escape / HEIGHT;
}

void
scroll_callback(GLFWwindow * window, double xoffset, double yoffset)
{
	float x_delta = x_end - x_start;
	float y_delta = y_end - y_start;

	if (yoffset > 0 && (x_delta < ZOOMLIMIT || y_delta < ZOOMLIMIT)) return;

	if (yoffset > 0) {
		x_start += x_delta / 8;
		x_end -= x_delta / 8;

		y_start += y_delta / 8;
		y_end -= y_delta / 8;
	}
	else {
		x_start -= x_delta / 4;
		x_end += x_delta / 4;

		y_start -= y_delta / 4;
		y_end += y_delta / 4;
	}

	//printf("start/end: %f, %f, %f, %f\n", x_start, x_end, y_start, y_end);

	x_step = (x_end - x_start) / WIDTH;
	y_step = (y_end - y_start) / HEIGHT;

	//printf("step: %f, %f", x_step, y_step);

	//double xpos, ypos;
	//glfwGetCursorPos(window, &xpos, &ypos);
	//printf("[%f, %f] %f, %f\n", xpos, ypos, xoffset, yoffset);

	update = true;
}

int
main(int argc, char * argv[])
{
	HMODULE hLib = LoadLibraryA("./libfractal.dll");
	if (hLib == NULL) {
		printf("Failed to load libfractal.dll\n");
		exit(EXIT_FAILURE);
	}

	void(__cdecl * fractal_get_colors) (HCMATRIX hCmatrix, struct FractalProperties * fp) = GetProcAddress(hLib, "fractal_get_colors");
	void(__cdecl * fractal_get_colors_th) (HCMATRIX hCmatrix, struct FractalProperties * fp, int num_threads) = GetProcAddress(hLib, "fractal_get_colors_th");
	void(__cdecl * fractal_avxf_get_colors_th) (HCMATRIX hCmatrix, struct FractalProperties * fp, int num_threads) = GetProcAddress(hLib, "fractal_avxf_get_colors_th");
	void(__cdecl * fractal_avxf_get_colors) (HCMATRIX hCmatrix, struct FractalProperties * fp) = GetProcAddress(hLib, "fractal_avxf_get_colors");
	float(__cdecl * fractal_cmatrix_max) (HCMATRIX hCmatrix) = GetProcAddress(hLib, "fractal_cmatrix_max");
	HCMATRIX(__cdecl * fractal_cmatrix_create) (int ROWS, int COLS) = GetProcAddress(hLib, "fractal_cmatrix_create");
	float * (__cdecl * fractal_cmatrix_value) (HCMATRIX hCmatrix, int row, int col) = GetProcAddress(hLib, "fractal_cmatrix_value");
	void(__cdecl * fractal_value_to_color) (float * r, float * g, float * b, enum Color color) = GetProcAddress(hLib, "fractal_value_to_color");

	if (!glfwInit()) {
		exit(EXIT_FAILURE);
	}
	glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 2);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 0);
	GLFWwindow * window = glfwCreateWindow(WIDTH, HEIGHT, "Fractals", NULL, NULL);
	if (!window) {
		glfwTerminate();
		exit(EXIT_FAILURE);
	}
	glfwMakeContextCurrent(window);
	glfwSwapInterval(1);

	glfwSetScrollCallback(window, scroll_callback);

	// Initialize the color matrix
	HCMATRIX hCmatrix = fractal_cmatrix_create(WIDTH, HEIGHT);
	view_reset();

	struct FractalProperties fp = {
		.x_start = x_start,
		.x_end = x_end,
		.y_start = y_start,
		.y_end = y_end,
		.width = WIDTH,
		.height = HEIGHT,
		.frac = FRAC_Z2,
		.mode = MODE_JULIA,
		.c_real = 0,
		.c_imag = 0,
		.R = R_escape,
		.max_iterations = MAX_ITERATIONS,
	};

	enum ColorFunction cf = CF_AVX;
	enum Color color = COLOR_JET;

	float r, g, b;

	bool first_run = true;
	clock_t time_since_color_change = 0;
	clock_t time_since_frac_change = 0;
	clock_t time_since_mode_change = 0;


	printf("Controls:\n"
		   "Q:                  Quit\n"
		   "R:                  Reset view\n"
		   "W, A, S, D:         Pan around\n"
		   "Scroll mousewheel:  Zoom in/out\n"
		   "Left/Right arrow:   Rotate (Julia mode)\n"
		   "1, 2, 3, 4: 	    Select solver method: Default, Threaded, AVX, AVX Threaded\n"
		   "C:                  Cycle colors\n"
		   "F:                  Cycle fractals\n"
		   "M:                  Cycle modes (Julia/Mandelbrot)\n\n\n");

	float counter = 0;
	float step = 0.1;
	while (!glfwWindowShouldClose(window)) {
		if (first_run) {
			update = true;
			first_run = false;
		}
		else {
			update = false;
		}

		glfwPollEvents();

		// PAN KEYS
		if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS) {
			x_start += x_step * PAN_STEP;
			x_end += x_step * PAN_STEP;
			update = true;
		}
		else if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS) {
			x_start -= x_step * PAN_STEP;
			x_end -= x_step * PAN_STEP;
			update = true;
		}
		else if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS) {
			y_start -= y_step * PAN_STEP;
			y_end -= y_step * PAN_STEP;
			update = true;
		}
		else if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS) {
			y_start += y_step * PAN_STEP;
			y_end += y_step * PAN_STEP;
			update = true;
		}
		// ROTATE KEYS
		else if (glfwGetKey(window, GLFW_KEY_RIGHT) == GLFW_PRESS) {
			counter += 0.1;
			update = true;
		}
		else if (glfwGetKey(window, GLFW_KEY_LEFT) == GLFW_PRESS) {
			counter -= 0.1;
			update = true;
		}
		// RESET KEY
		else if (glfwGetKey(window, GLFW_KEY_R) == GLFW_PRESS) {
			view_reset();
			update = true;
		}
		// QUIT KEY
		else if (glfwGetKey(window, GLFW_KEY_Q) == GLFW_PRESS) {
			break;
		}
		// COLOR FUNCTION KEYS
		else if (glfwGetKey(window, GLFW_KEY_1) == GLFW_PRESS) {
			cf = CF_DEFAULT;
			printf("Switching to CF_DEFAULT\n");
			update = true;
		}
		else if (glfwGetKey(window, GLFW_KEY_2) == GLFW_PRESS) {
			cf = CF_THREADED;
			printf("Switching to CF_THREADED\n");
			update = true;
		}
		else if (glfwGetKey(window, GLFW_KEY_3) == GLFW_PRESS) {
			cf = CF_AVX;
			printf("Switching to CF_AVX\n");
			update = true;
		}
		else if (glfwGetKey(window, GLFW_KEY_4) == GLFW_PRESS) {
			cf = CF_AVX_THREADED;
			printf("Switching to avx CF_AVX_THREADED\n");
			update = true;
		}
		else if (glfwGetKey(window, GLFW_KEY_C) == GLFW_PRESS) {
			if ((clock() - time_since_color_change) > 0.4f * CLOCKS_PER_SEC) {
				// only cycle the color mode if it has been changed at least .4 seconds ago
				time_since_color_change = clock();
				color = (color + 1) % COLOR_NUM_ENTRIES;
				update = true;
			}
		}
		else if (glfwGetKey(window, GLFW_KEY_F) == GLFW_PRESS) {
			if ((clock() - time_since_frac_change) > 0.4f * CLOCKS_PER_SEC) {
				// only cycle the fractal mode if it has been changed at least .4 seconds ago
				time_since_frac_change = clock();
				fp.frac = (fp.frac + 1) % FRAC_NUM_ENTRIES;
				update = true;
			}
		}
		else if (glfwGetKey(window, GLFW_KEY_M) == GLFW_PRESS) {
			if ((clock() - time_since_mode_change) > 0.4f * CLOCKS_PER_SEC) {
				// only cycle the mode if it has been changed at least .4 seconds ago
				time_since_mode_change = clock();
				fp.mode = (fp.mode + 1) % MODE_NUM_ENTRIES;
				update = true;
			}
		}


		//Setup View
		float ratio;
		int width, height;
		glfwGetFramebufferSize(window, &width, &height);
		ratio = width / (float)height;
		glViewport(0, 0, width, height);
		glClear(GL_COLOR_BUFFER_BIT);

		// c = 0.7885 * exp(j*x)
		//   = 0.7885 * (cos(x) + j * sin(x))
		//   = 0.7885 * cos(x) + j * 0.7885 * sin(x)
		// x = 0..2*PI
		//counter = 61.2;
		fp.c_real = 0.7885 * cosf(counter / (2 * PI));
		fp.c_imag = 0.7885 * sinf(counter / (2 * PI));
		fp.x_start = x_start;
		fp.x_end = x_end;
		fp.y_start = y_start;
		fp.y_end = y_end;

		//counter += 0.2;

		// Compute the color matrix
		switch (cf) {
		case CF_DEFAULT:
			fractal_get_colors(hCmatrix, &fp); break;
		case CF_THREADED:
			fractal_get_colors_th(hCmatrix, &fp, 8);  break;
		case CF_AVX:
			fractal_avxf_get_colors(hCmatrix, &fp); break;
		case CF_AVX_THREADED:
			fractal_avxf_get_colors_th(hCmatrix, &fp, 8); break;
		}

		// Find the maximum color value in the matrix
		//float max_color = fractal_cmatrix_max(hCmatrix);

		if (update) {
			// Draw the fractal
			glBegin(GL_POINTS);
			for (int row = 0; row < WIDTH; ++row) {
				for (int col = 0; col < HEIGHT; ++col) {
					fractal_value_to_color(&r, &g, &b, (int)*fractal_cmatrix_value(hCmatrix, row, col), color);

					glColor3f(r, g, b);
					glVertex2f(
						(((float)row) / WIDTH - 0.5) * 2,
						(((float)col) / HEIGHT - 0.5) * 2
					);
				}
			}
			glEnd();

			//Swap buffer and check for events
			glfwSwapBuffers(window);
		}
	}

	glfwDestroyWindow(window);
	glfwTerminate();
	exit(EXIT_SUCCESS);
}
