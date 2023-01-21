/** Function declarations */
bool fractal_escape_magnitude_check(float z_real, float z_imag, float R);

void fractal_z2(float * result_real, float * result_imag, const float z_real, const float z_imag, const float c_real, const float c_imag);
void fractal_z3(float * result_real, float * result_imag, const float z_real, const float z_imag, const float c_real, const float c_imag);
void fractal_z4(float * result_real, float * result_imag, const float z_real, const float z_imag, const float c_real, const float c_imag);
void fractal_zconj2(float * result_real, float * result_imag, const float z_real, const float z_imag, const float c_real, const float c_imag);
void fractal_zconj3(float * result_real, float * result_imag, const float z_real, const float z_imag, const float c_real, const float c_imag);
void fractal_zconj4(float * result_real, float * result_imag, const float z_real, const float z_imag, const float c_real, const float c_imag);
void fractal_zabs2(float * result_real, float * result_imag, const float z_real, const float z_imag, const float c_real, const float c_imag);
void fractal_zabs3(float * result_real, float * result_imag, const float z_real, const float z_imag, const float c_real, const float c_imag);
void fractal_zabs4(float * result_real, float * result_imag, const float z_real, const float z_imag, const float c_real, const float c_imag);
void fractal_magnet(float * result_real, float * result_imag, const float z_real, const float z_imag, const float c_real, const float c_imag);
void fractal_z2_z(float * result_real, float * result_imag, const float z_real, const float z_imag, const float c_real, const float c_imag);


/** Fractal functions */
enum FC_Fractal {
    FC_FRAC_Z2,     /**< z^2 + c */
    FC_FRAC_Z3,     /**< z^3 + c */
    FC_FRAC_Z4,     /**< z^4 + c */
    FC_FRAC_ZCONJ2, /**< (conj(z))^2 + c */
    FC_FRAC_ZCONJ3, /**< (conj(z))^3 + c */
    FC_FRAC_ZCONJ4, /**< (conj(z))^4 + c */
    FC_FRAC_ZABS2,  /**< (abs(z_real) + abs(c_real)*j)^2 + c */
    FC_FRAC_ZABS3,  /**< (abs(z_real) + abs(c_real)*j)^3 + c */
    FC_FRAC_ZABS4,  /**< (abs(z_real) + abs(c_real)*j)^4 + c */
    FC_FRAC_MAGNET, /**< [(z^2 + c - 1)/(2z + c - 2)]^2 */
    FC_FRAC_Z2_Z,   /**< z^2 + c/z */
    FC_FRAC_NUM_ENTRIES,
};

/** Fractal modes */
enum FC_Mode {
    FC_MODE_MANDELBROT,
    FC_MODE_JULIA,
    FC_MODE_LYAPUNOV,
    FC_MODE_FLAMES,
    FC_MODE_NUM_ENTRIES,
};

/** Color modes to convert CMATRIX values to */
enum FC_Color {
    FC_COLOR_ULTRA,
    FC_COLOR_MONOCHROME,
    FC_COLOR_TRI,
    FC_COLOR_JET,
    FC_COLOR_LAVENDER,
    FC_COLOR_BINARY,
    FC_COLOR_NUM_ENTRIES
};

/* types */
struct Flame {
    int width;              /**< The width in pixels of the flame */
    int height;             /**< The height in pixels of the flame */
    int num_chaos_games;    /**< The number of chaos games to run */
    int chaos_game_length;  /**< The number of iterations in a chaos game */
    int supersample;        /**< Reduce noise by computing the fractal flame in a higher resolution */
    float gamma;            /**< Gamma correction */
    char * savename;        /**< Save name for the resulting image */
};

struct FractalProperties {
    float x_start;          /**< Lower bound of the real axis in the complex plane */
    float x_end;            /**< Upper bound of the real axis in the complex plane */
    float y_start;          /**< Lower bound of the imaginary axis in the complex plane */
    float y_end;            /**< Upper bound of the imaginary axis in the complex plane */
    enum FC_Fractal frac;   /**< The fractal function which will be iterated */
    enum FC_Mode mode;      /**< The fractal mode */
    enum FC_Color color;    /**< The color mode to render the fractals (required for cuda) */
    float c_real;           /**< The real part of the c parameter in the fractal function */
    float c_imag;           /**< The imaginary part of the c parameter in the fractal function */
    float R;                /**< Escape radius */
    int max_iterations;     /**< Maximum number of times the fractal function will be iterated */
    char * sequence;        /**< Lyapunov sequence */
    size_t sequence_length; /**< Lyapunov sequence length */
    struct Flame flame;     /**< Fractal flame properties
                             * @warning @c HCMATRIX is not used by fractal flames, and the image is immediately saved. */
    float _x_step;
    float _y_step;
};
