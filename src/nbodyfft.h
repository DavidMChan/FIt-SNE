#ifndef NBODYFFT_H
#define NBODYFFT_H

#ifdef _WIN32
#include "winlibs/fftw3.h"
#else
#include <fftw3.h>
#endif
#include <complex>

using namespace std;

typedef float (*kernel_type)(float, float);

typedef float (*kernel_type_2d)(float, float, float, float);

void precompute_2d(float x_max, float x_min, float y_max, float y_min, int n_boxes, int n_interpolation_points,
                   kernel_type_2d kernel, float *box_lower_bounds, float *box_upper_bounds, float *y_tilde_spacings,
                   float *y_tilde, float *x_tilde, complex<float> *fft_kernel_tilde);

void n_body_fft_2d(int N, int n_terms, float *xs, float *ys, float *chargesQij, int n_boxes,
                   int n_interpolation_points, float *box_lower_bounds, float *box_upper_bounds,
                   float *y_tilde_spacings, complex<float> *fft_kernel_tilde, float *potentialQij, const float *denominator, unsigned int nthreads);

void interpolate(int n_interpolation_points, int N, const float *y_in_box, const float *y_tilde_spacings,
                 float *interpolated_values, const float *denominator);

#endif
