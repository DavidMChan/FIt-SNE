#ifndef NBODYFFT_H
#define NBODYFFT_H

#ifdef _WIN32
#include "winlibs/fftw3.h"
#else
#include <fftw3.h>
#endif
#include <complex>
#include "common.h"

using namespace std;

typedef float (*kernel_type)(float, float);

typedef float (*kernel_type_2d)(float, float, float, float);

void precompute_2d(float x_max, float x_min, float y_max, float y_min, int n_boxes, int n_interpolation_points,
                   kernel_type_2d kernel, float *box_lower_bounds, float *box_upper_bounds, float *y_tilde_spacings,
                   float *y_tilde, float *x_tilde, complex<float> *fft_kernel_tilde);

void n_body_fft_2d(
    int N, 
    int n_terms, 
    int n_boxes,
    int n_interpolation_points, 
    complex<float> *fft_kernel_tilde,
    const float *denominator,
    int n_total_boxes,
    int total_interpolation_points,
    float coord_min,
    float box_width,
    int n_fft_coeffs_half,
    int n_fft_coeffs,
    thrust::device_vector<int> &point_box_idx_device, 
    thrust::device_vector<float> &x_in_box_device,
    thrust::device_vector<float> &y_in_box_device,
    thrust::device_vector<float> &xs_device,
    thrust::device_vector<float> &ys_device,
    thrust::device_vector<float> &box_lower_bounds_device,
    thrust::device_vector<float> &y_tilde_spacings_device,
    thrust::device_vector<float> &denominator_device,
    thrust::device_vector<float> &y_tilde_values,
    thrust::device_vector<float> &all_interpolated_values_device,
    thrust::device_vector<float> &output_values,
    thrust::device_vector<int> &all_interpolated_indices,
    thrust::device_vector<int> &output_indices,
    thrust::device_vector<float> &w_coefficients_device,
    thrust::device_vector<float> &chargesQij_device,
    thrust::device_vector<float> &x_interpolated_values_device,
    thrust::device_vector<float> &y_interpolated_values_device,
    thrust::device_vector<float> &potentialsQij_device);

void interpolate(int n_interpolation_points, int N, const float *y_in_box, const float *y_tilde_spacings,
                 float *interpolated_values, const float *denominator);

float* get_ntime();

#endif
