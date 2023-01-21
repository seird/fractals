#define COLORMAP_SIZE 16


void
color_jet(global uchar * image, global uchar * M, size_t n)
{
    const uchar colormap[16][3] = {
        {  0,     0,   191},
        {  0,     0,   255},
        {  0,    63,   255},
        {  0,   127,   255},
        {  0,   191,   255},
        {  0,   255,   255},
        { 63,   255,   191},
        {127,   255,   127},
        {191,   255,    63},
        {255,   255,     0},
        {255,   191,     0},
        {255,   127,     0},
        {255,    63,     0},
        {255,     0,     0},
        {191,     0,     0},
        {127,     0,     0},
    };

    if (M[n] > 0) {
        image[n * 3] = colormap[M[n] % 16][0];
        image[n * 3 + 1] = colormap[M[n] % 16][1];
        image[n * 3 + 2] = colormap[M[n] % 16][2];
    } else {
        image[n * 3] = image[n * 3 + 1] = image[n * 3 + 2] = 0;
    }
}


void
color_monochrome(global uchar * image, global uchar * M, size_t n)
{
    const uchar colormap[15][3] = {
        { 50,  50,  50},
        { 60,  60,  60},
        { 70,  70,  70},
        { 80,  80,  80},
        { 90,  90,  90},
        {100, 100, 100},
        {110, 110, 110},
        {120, 120, 120},
        {130, 130, 130},
        {140, 140, 140},
        {150, 150, 150},
        {160, 160, 160},
        {170, 170, 170},
        {180, 180, 180},
        {190, 190, 190},
    };

    if (M[n] > 0) {
        image[n * 3] = colormap[M[n] % 15][0];
        image[n * 3 + 1] = colormap[M[n] % 15][1];
        image[n * 3 + 2] = colormap[M[n] % 15][2];
    } else {
        image[n * 3] = image[n * 3 + 1] = image[n * 3 + 2] = 0;
    }
}


void
color_tri(global uchar * image, global uchar * M, size_t n)
{
    const uchar colormap[3][3] = {
        {255, 150, 150},
        {150, 255, 150},
        {150, 150, 255},
    };

    if (M[n] > 0) {
        image[n * 3]= colormap[M[n] % 3][0];
        image[n * 3 + 1]= colormap[M[n] % 3][1];
        image[n * 3 + 2]= colormap[M[n] % 3][2];
    } else {
        image[n * 3] = image[n * 3 + 1] = image[n * 3 + 2] = 0;
    }
}



void
color_ultra(global uchar * image, global uchar * M, size_t n)
{
    const uchar colormap[16][3] = {
        { 66,  30,  15},
        { 25,   7,  26},
        {  9,   1,  47},
        {  4,   4,  73},
        {  0,   7, 100},
        { 12,  44, 138},
        { 24,  82, 177},
        { 57, 125, 209},
        {134, 181, 229},
        {211, 236, 248},
        {241, 233, 191},
        {248, 201,  95},
        {255, 170,   0},
        {204, 128,   0},
        {153,  87,   0},
        {106,  52,   3},
    };

    if (M[n] > 0) {
        image[n * 3] = colormap[M[n] % 16][0];
        image[n * 3 + 1] = colormap[M[n] % 16][1];
        image[n * 3 + 2] = colormap[M[n] % 16][2];
    } else {
        image[n * 3] = image[n * 3 + 1] = image[n * 3 + 2] = 0;
    }
}


void
color_lavender(global uchar * image, global uchar * M, size_t n) {
    const uchar colormap[34][3] = {
       { 69, 147, 254},
       {101, 154, 214},
       {108, 122, 211},
       {114,  89, 203},
       {119,  80, 183},
       {123, 134, 163},
       {128,  51, 186},
       {132,  88, 151},
       {136, 116, 153},
       {140, 126, 130},
       {144, 145, 111},
       {149,  56, 151},
       {153, 105, 124},
       {158,  23, 131},
       {162, 112, 117},
       {167,  60,  93},
       {172,  42,  83},
       {177,  44, 111},
       {182,  49,  86},
       {187, 163,   7},
       {193,  33,  57},
       {198, 188,  29},
       {204,  75,  91},
       {210,  48,  73},
       {216,  32,  85},
       {221, 118, 128},
       {226, 207,   0},
       {231,  68,  85},
       {235, 169, 169},
       {239, 147, 159},
       {243,  67,  93},
       {246, 179, 193},
       {250,  83, 103},
       {253, 224, 226}
    };

    if (M[n] > 0) {
        image[n * 3] = colormap[M[n] % 34][0];
        image[n * 3 + 1] = colormap[M[n] % 34][1];
        image[n * 3 + 2] = colormap[M[n] % 34][2];
    } else {
        image[n * 3] = image[n * 3 + 1] = image[n * 3 + 2] = 0;
    }
}


void
color_binary(global uchar * image, global uchar * M, size_t n)
{
    image[n * 3] = image[n * 3 + 1] = image[n * 3 + 2] = 255*(M[n] > 0);
}


__kernel void
colormap(__global uchar * image, __global uchar * M, const int width, const int height, const enum FC_Color color)
{
    size_t x = get_global_id(0);
    size_t y = get_global_id(1);

    size_t n = y * width + x;

    switch (color) {
        case FC_COLOR_ULTRA:
            color_ultra(image, M, n);
            break;
        case FC_COLOR_MONOCHROME:
            color_monochrome(image, M, n);
            break;
        case FC_COLOR_TRI:
            color_tri(image, M, n);
            break;
        case FC_COLOR_JET:
            color_jet(image, M, n);
            break;
        case FC_COLOR_LAVENDER:
            color_lavender(image, M, n);
            break;
        case FC_COLOR_BINARY:
            color_binary(image, M, n);
            break;
        default:
            break;
    }
}
