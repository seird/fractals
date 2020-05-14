#include <windows.h>

#include <stdint.h>
#include <stdbool.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>

#include <GLFW\glfw3.h>
#include "fractal_color.h"


#define PI 3.14159f
#define R_escape 2.0f

#define HEIGHT 1000
#define WIDTH 1000

#define ARROW_STEP 100


float x_start = -R_escape;
float x_end = R_escape;

float y_start = -R_escape;
float y_end = R_escape;

float x_step = 2 * R_escape / WIDTH;
float y_step = 2 * R_escape / HEIGHT;


void
scroll_callback(GLFWwindow * window, double xoffset, double yoffset)
{
	float x_delta = x_end - x_start;
	float y_delta = y_end - y_start;

	if (yoffset > 0 && (x_delta < 0.00001 || y_delta < 0.00001)) return;

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

	printf("start/end: %f, %f, %f, %f\n", x_start, x_end, y_start, y_end);

	x_step = (x_end - x_start) / WIDTH;
	y_step = (y_end - y_start) / HEIGHT;

	printf("step: %f, %f", x_step, y_step);

	double xpos, ypos;
	glfwGetCursorPos(window, &xpos, &ypos);
	printf("[%f, %f] %f, %f\n", xpos, ypos, xoffset, yoffset);
}

int
main(int argc, char * argv[])
{
	
	int max_iterations = 500;

	HMODULE hLib = LoadLibraryA("./libfractal.dll");
	if (hLib == NULL) {
		printf("Failed to load libfractal.dll\n");
		exit(EXIT_FAILURE);
	}

	void(__cdecl * fractal_get_colors) (HCMATRIX hCmatrix, struct FractalProperties * fp) = GetProcAddress(hLib, "fractal_get_colors");
	void(__cdecl * fractal_get_colors_th) (HCMATRIX hCmatrix, struct FractalProperties * fp, int num_threads) = GetProcAddress(hLib, "fractal_get_colors_th");
	float(__cdecl * fractal_get_max_color) (HCMATRIX hCmatrix) = GetProcAddress(hLib, "fractal_get_max_color");
	HCMATRIX(__cdecl * fractal_cmatrix_create) (int ROWS, int COLS) = GetProcAddress(hLib, "fractal_cmatrix_create");
	float * (__cdecl * fractal_cmatrix_value) (HCMATRIX hCmatrix, int row, int col) = GetProcAddress(hLib, "fractal_cmatrix_value");
	
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

	struct FractalProperties fp = {
		.x_start = x_start,
		.x_step = x_step,
		.y_start = y_start,
		.y_step = y_step,
		.frac = FRAC_JULIA,
		.c_real = 0,
		.c_imag = 0,
		.R = R_escape,
		.max_iterations = max_iterations,
	};


	float r, g, b;

	float counter = 0;
	float step = 0.1;
	while (!glfwWindowShouldClose(window)) {
		if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS) {
			x_start -= x_step * ARROW_STEP;
			x_end -= x_step * ARROW_STEP;
		}
		else if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS) {
			x_start += x_step * ARROW_STEP;
			x_end += x_step * ARROW_STEP;
		}
		else if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS) {
			y_start -= y_step * ARROW_STEP;
			y_end -= y_step * ARROW_STEP;
		}
		else if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS) {
			y_start += y_step * ARROW_STEP;
			y_end += y_step * ARROW_STEP;
		}
		else if (glfwGetKey(window, GLFW_KEY_RIGHT) == GLFW_PRESS) {
			counter += 0.1;
		}
		else if (glfwGetKey(window, GLFW_KEY_LEFT) == GLFW_PRESS) {
			counter -= 0.1;
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
		fp.x_step = x_step;
		fp.y_step = y_step;
		fp.x_start = x_start;
		fp.y_start = y_start;
		//counter += step;

		//fractal_get_colors(hCmatrix, &fp);
		fractal_get_colors_th(hCmatrix, &fp, 12);
		float max_color = fractal_get_max_color(hCmatrix);

		glBegin(GL_POINTS);
		for (int row = 0; row < WIDTH; ++row) {
			for (int col = 0; col < HEIGHT; ++col) {
				r = g = b = *fractal_cmatrix_value(hCmatrix, row, col) / max_color;
				r *= 2.2;
				//g *= 3;
				b *= 2;

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
		glfwPollEvents();
	}

	glfwDestroyWindow(window);
	glfwTerminate;
	exit(EXIT_SUCCESS);
}
