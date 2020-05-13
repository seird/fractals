#include <windows.h>

#include <stdint.h>
#include <stdbool.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>

#include <GLFW\glfw3.h>

#include "fractal_color.h"


#define PI 3.14159


int
main(int argc, char * argv[])
{
	int height = 1000;
	int width = 1000;
	float R = 2;
	int max_iterations = 100;

	HMODULE hLib = LoadLibraryA("./libfractal.dll");
	if (hLib == NULL) {
		printf("Failed to load libfractal.dll\n");
		exit(EXIT_FAILURE);
	}

	void(__cdecl * fractal_get_colors_nc) (HCMATRIX hCmatrix, float x_start, float x_step, float y_start, float y_step, enum Fractal frac, float c_real, float c_imag, float R, int max_iterations) = GetProcAddress(hLib, "fractal_get_colors_nc");
	void(__cdecl * fractal_get_colors_th) (HCMATRIX hCmatrix, float x_start, float x_step, float y_start, float y_step, enum Fractal frac, float c_real, float c_imag, float R, int max_iterations, int num_threads) = GetProcAddress(hLib, "fractal_get_colors_th");
	float(__cdecl * fractal_get_max_color) (HCMATRIX hCmatrix) = GetProcAddress(hLib, "fractal_get_max_color");
	HCMATRIX(__cdecl * fractal_cmatrix_create) (int ROWS, int COLS) = GetProcAddress(hLib, "fractal_cmatrix_create");
	float * (__cdecl * fractal_cmatrix_value) (HCMATRIX hCmatrix, int row, int col) = GetProcAddress(hLib, "fractal_cmatrix_value");

	if (!glfwInit()) {
		exit(EXIT_FAILURE);
	}
	glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 2);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 0);
	GLFWwindow * window = glfwCreateWindow(width, height, "Fractals", NULL, NULL);
	if (!window) {
		glfwTerminate();
		exit(EXIT_FAILURE);
	}
	glfwMakeContextCurrent(window);
	glfwSwapInterval(1);

	int ROWS = height;
	int COLS = width;

	// Initialize the color matrix
	HCMATRIX hCmatrix = fractal_cmatrix_create(ROWS, COLS);

	float x_start = -R;
	float x_end = R;

	float y_start = -R;
	float y_end = R;

	float x_step = (x_end - x_start) / ROWS;
	float y_step = (y_end - y_start) / COLS;


	float c_real = 0;
	float c_imag = 0;


	float r, g, b;

	float counter = 0;
	float step = 0.1;
	while (!glfwWindowShouldClose(window)) {
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
		c_real = 0.7885 * cosf(counter / (2 * PI));
		c_imag = 0.7885 * sinf(counter / (2 * PI));
		counter += step;

		fractal_get_colors_nc(hCmatrix, x_start, x_step, y_start, y_step, FRAC_JULIA, c_real, c_imag, R, max_iterations);
		//fractal_get_colors_th(hCmatrix, x_start, x_step, y_start, y_step, FRAC_JULIA, c_real, c_imag, R, max_iterations, 12);
		float max_color = fractal_get_max_color(hCmatrix);

		glBegin(GL_POINTS);
		for (int row = 0; row < ROWS; ++row) {
			for (int col = 0; col < COLS; ++col) {
				r = g = b = *fractal_cmatrix_value(hCmatrix, row, col) / max_color;
				r *= 3;
				//g *= 3;
				b *= 3;

				glColor3f(r, g, b);
				glVertex2f(
					(((float)row) / ROWS - 0.5) * 2,
					(((float)col) / COLS - 0.5) * 2
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
