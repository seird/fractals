#include <stdio.h>
#include <stdlib.h>
#include <complex.h>
#include <math.h>
#include <time.h>

#define CL_TARGET_OPENCL_VERSION 300
#include <CL/cl.h>

#include "../include/fractal_color.h"
#include "../include/fractal_opencl.h"

#include "kernels/kernels.h"


#define MAX_SOURCE_SIZE (0x100000)
#define MAX_LYAP_SIZE 0xFF


static cl_uint num_sources;
static char ** sourcestrs;
static size_t * sourcelengths;

static cl_int ret;
static cl_context context;
static cl_device_id device_id = NULL;
static cl_program program;
static cl_command_queue command_queue;
static cl_kernel kernel_julia;
static cl_kernel kernel_mandelbrot;
static cl_kernel kernel_newton;
static cl_kernel kernel_lyapunov;
static cl_kernel kernel_colormap;
static cl_mem m_mem_obj;
static cl_mem fp_mem_obj;
static cl_mem sequence_mem_obj;
static cl_mem image_mem_obj;

static int image_width;
static int image_height;


bool
fractal_opencl_init(int width, int height)
{
    image_width = width;
    image_height = height;

    // Load the kernel source code in the array sourcestrs	
    load_kernels(&num_sources, &sourcestrs, &sourcelengths);

    // Get platform and device information
    cl_platform_id platform_id = NULL;
    cl_uint ret_num_devices;
    cl_uint ret_num_platforms;
    ret = clGetPlatformIDs(1, &platform_id, &ret_num_platforms);
    ret = clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_ALL, 1, &device_id, &ret_num_devices);

    // Create an OpenCL context
    context = clCreateContext(NULL, 1, &device_id, NULL, NULL, &ret);

    // Create a program from the kernel source
	program = clCreateProgramWithSource(context, num_sources, (const char**)sourcestrs, (const size_t *)sourcelengths, &ret);
    if (ret != CL_SUCCESS) {
        fprintf(stderr, "error creating program: %d\n", ret);
        return false;
    }

    // Build the program
    ret = clBuildProgram(program, 1, &device_id, NULL, NULL, NULL);
    if (ret != CL_SUCCESS) {
        printf("failed to build program: %d\n", ret);
    }

    // Create the OpenCL kernels
    kernel_julia = clCreateKernel(program, "julia", &ret);
    kernel_mandelbrot = clCreateKernel(program, "mandelbrot", &ret);
    kernel_newton = clCreateKernel(program, "newton", &ret);
    kernel_lyapunov = clCreateKernel(program, "lyapunov", &ret);
    kernel_colormap = clCreateKernel(program, "colormap", &ret);

    // Create a command queue
    command_queue = clCreateCommandQueueWithProperties(context, device_id, NULL, &ret);

    // Create memory buffers for the kernel inputs
    m_mem_obj = clCreateBuffer(context, CL_MEM_READ_WRITE, width * height * sizeof(cl_uint), NULL, &ret);
    fp_mem_obj = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(struct FractalProperties), NULL, &ret);
    sequence_mem_obj = clCreateBuffer(context, CL_MEM_READ_ONLY, MAX_LYAP_SIZE * sizeof(char), NULL, &ret);
    image_mem_obj = clCreateBuffer(context, CL_MEM_READ_WRITE, 3 * width * height * sizeof(uint8_t), NULL, &ret);

    return true;
}


void
fractal_opencl_clean()
{
    if (num_sources) {
        free(sourcelengths);
        free(sourcestrs);
    }
    sourcestrs = NULL;
    sourcelengths = NULL;
    num_sources = 0;

    ret = clFlush(command_queue);
    ret |= clFinish(command_queue);
    ret |= clReleaseKernel(kernel_julia);
    ret |= clReleaseKernel(kernel_mandelbrot);
    ret |= clReleaseKernel(kernel_newton);
    ret |= clReleaseKernel(kernel_lyapunov);
    ret |= clReleaseKernel(kernel_colormap);
    ret |= clReleaseProgram(program);
    ret |= clReleaseMemObject(m_mem_obj);
    ret |= clReleaseMemObject(fp_mem_obj);
    ret |= clReleaseMemObject(sequence_mem_obj);
    ret |= clReleaseMemObject(image_mem_obj);
    ret |= clReleaseCommandQueue(command_queue);
    ret |= clReleaseContext(context);
    if (ret != CL_SUCCESS) {
        printf("opencl cleanup failed: %d\n", ret);
    }
}


static bool
do_kernel_fractal(uint8_t * image, struct FractalProperties * fp)
{
    /* ---------------------------------------------------------- */
    /*                     Fratal Computation                     */
    /* ---------------------------------------------------------- */

    // Copy the inputs to their respective memory buffers
    ret |= clEnqueueWriteBuffer(command_queue, fp_mem_obj, CL_TRUE, 0, sizeof(struct FractalProperties), fp, 0, NULL, NULL);
    if (fp->mode == FC_MODE_LYAPUNOV) {
        ret |= clEnqueueWriteBuffer(command_queue, sequence_mem_obj, CL_TRUE, 0, sizeof(char) * fp->sequence_length, fp->sequence, 0, NULL, NULL);
    }
    if (ret != CL_SUCCESS) {
        printf("failed to copy data from the host to the device: %d\n", ret);
        return false;
    }

    cl_kernel kernel_mode = NULL;
    switch (fp->mode) {
        case FC_MODE_JULIA:
            kernel_mode = kernel_julia;
            break;
        case FC_MODE_MANDELBROT:
            kernel_mode = kernel_mandelbrot;
            break;
        case FC_MODE_NEWTON:
            kernel_mode = kernel_newton;
            break;
        case FC_MODE_LYAPUNOV:
            kernel_mode = kernel_lyapunov;
            break;
        default:
            kernel_mode = kernel_julia;
    }

    // Set the arguments of the kernel
    ret = clSetKernelArg(kernel_mode, 0, sizeof(cl_mem), (void *)&m_mem_obj);
    ret |= clSetKernelArg(kernel_mode, 1, sizeof(cl_int), (void *)&image_width);
    ret |= clSetKernelArg(kernel_mode, 2, sizeof(cl_int), (void *)&image_height);
    ret |= clSetKernelArg(kernel_mode, 3, sizeof(cl_mem), (void *)&fp_mem_obj);
    if (fp->mode == FC_MODE_LYAPUNOV) {
        ret |= clSetKernelArg(kernel_mode, 4, sizeof(cl_mem), (void *)&sequence_mem_obj);
        ret |= clSetKernelArg(kernel_mode, 5, sizeof(cl_int), (void *)&fp->sequence_length);
    }
    if (ret != CL_SUCCESS) {
        printf("failed to set the fractal kernel arguments: %d\n", ret);
        return false;
    }

    // Execute the OpenCL fractal kernel
    size_t global_item_size[2] = {image_width, image_height}; // Process the entire width*height plane
    // size_t local_item_size = 64; // Divide work items into groups of 64
    clEnqueueNDRangeKernel(command_queue, kernel_mode, 2, NULL, global_item_size, NULL, 0, NULL, NULL);

    return true;
}


static bool
do_kernel_colormap(uint8_t * image, struct FractalProperties * fp)
{
    /* ---------------------------------------------------------- */
    /*                   Colormap Computation                     */
    /* ---------------------------------------------------------- */

    // Flush the command queue
    ret = clFlush(command_queue);

    // Set the arguments of the kernel
    ret = clSetKernelArg(kernel_colormap, 0, sizeof(cl_mem), (void *)&image_mem_obj);
    ret |= clSetKernelArg(kernel_colormap, 1, sizeof(cl_mem), (void *)&m_mem_obj);
    ret |= clSetKernelArg(kernel_colormap, 2, sizeof(cl_int), (void *)&image_width);
    ret |= clSetKernelArg(kernel_colormap, 3, sizeof(cl_int), (void *)&image_height);
    ret |= clSetKernelArg(kernel_colormap, 4, sizeof(cl_GLenum), (void *)&fp->color);
    if (ret != CL_SUCCESS) {
        printf("failed to set the colormap kernel arguments: %d\n", ret);
        return false;
    }

    // Execute the OpenCL colormap kernel
    size_t global_item_size[2] = {image_width, image_height}; // Process the entire width*height plane
    clEnqueueNDRangeKernel(command_queue, kernel_colormap, 2, NULL, global_item_size, NULL, 0, NULL, NULL);

    return true;
}

void
fractal_opencl_get_colors(uint8_t * image, struct FractalProperties * fp)
{
    do_kernel_fractal(image, fp);
    do_kernel_colormap(image, fp);

    // Read the memory buffer image on the device to the local variable image
    ret = clEnqueueReadBuffer(command_queue, image_mem_obj, CL_TRUE, 0, 3 * image_width * image_height * sizeof(uint8_t), image, 0, NULL, NULL);
    if (ret != CL_SUCCESS) {
        printf("failed to copy data from the device to the host: %d\n", ret);
    }
}


#if !defined(SHARED) && !defined(STATIC)
int
main()
{
    /* ----------- INPUT PARAMETERS ----------- */
    float c_real = 1.0f;
	float c_imag = 0.0f;
    float _Complex c = c_real + c_imag*I;
    float R = ceilf(cabsf(c)) + 1;
    
    int height = 3*16*24;
    int width = 3*16*24;
    
    float aspect_ratio = (float)width/height;

    float x_start = -2.5f;
    float x_end   =  1.0f;

    float y_start = -2.0f;
    float y_end = 1.0f;    

    int max_iterations = 1000;

    enum FC_Mode mode = FC_MODE_LYAPUNOV;
    enum FC_Fractal fractal = FC_FRAC_N_SIN1;
    enum FC_Color color = FC_COLOR_ULTRA;


    struct FractalProperties fp = {
        .x_start = x_start,
        .x_end = x_end,
        .y_start = y_start,
        .y_end = y_end,

        .frac = fractal,
        .mode = mode,
        .color = color,

        .c_real = c_real,
        .c_imag = c_imag,
        .R = R,
        .max_iterations = max_iterations,

        .sequence = "AABAB",
        .sequence_length = sizeof("AABAB"),
    };
    /* ---------------------------------------- */

    if (!fractal_opencl_init(width, height)) {
        printf("Failed to initialize the opencl environment\n");
        return 0;
    }

    uint8_t * image = fractal_opencl_image_create(width, height);

    float time_start = (float) clock()/CLOCKS_PER_SEC;
    for (int i=0; i<1; ++i) {
        fp.c_real+=0.01;
        fractal_opencl_get_colors(image, &fp);
    }

    float time_elapsed = (float) clock()/CLOCKS_PER_SEC - time_start;
    printf("%ld ms total\n\n", (long)(1000*time_elapsed));

    // Save the result
    fractal_opencl_image_save(image, image_width, image_height, "fractal-opencl.png");

    // Clean up
    fractal_opencl_clean();
    fractal_opencl_image_free(image);
    
    printf("Done\n");

    return 0;
}
#endif
