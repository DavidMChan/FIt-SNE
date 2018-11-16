#include "winlibs/stdafx.h"

#include "parallel_for.h"
#include "time_code.h"
#include "nbodyfft.h"
#include <cufft.h>
#include "common.h"
#include "include/util/cuda_utils.h"
#include <thrust/complex.h>
#include "include/util/matrix_broadcast_utils.h"

#define cufftSafeCall(err)  __cufftSafeCall(err, __FILE__, __LINE__)

static const char *_cudaGetErrorEnum(cufftResult error)
{
    switch (error)
    {
        case CUFFT_SUCCESS:
            return "CUFFT_SUCCESS";

        case CUFFT_INVALID_PLAN:
            return "CUFFT_INVALID_PLAN";

        case CUFFT_ALLOC_FAILED:
            return "CUFFT_ALLOC_FAILED";

        case CUFFT_INVALID_TYPE:
            return "CUFFT_INVALID_TYPE";

        case CUFFT_INVALID_VALUE:
            return "CUFFT_INVALID_VALUE";

        case CUFFT_INTERNAL_ERROR:
            return "CUFFT_INTERNAL_ERROR";

        case CUFFT_EXEC_FAILED:
            return "CUFFT_EXEC_FAILED";

        case CUFFT_SETUP_FAILED:
            return "CUFFT_SETUP_FAILED";

        case CUFFT_INVALID_SIZE:
            return "CUFFT_INVALID_SIZE";

        case CUFFT_UNALIGNED_DATA:
            return "CUFFT_UNALIGNED_DATA";
    }

    return "<unknown>";
}

inline void __cufftSafeCall(cufftResult err, const char *file, const int line)
{
    if( CUFFT_SUCCESS != err) {
		fprintf(stderr, "CUFFT error in file '%s', line %d\n %s\nerror %d: %s\nterminating!\n",__FILE__, __LINE__,err, \
									_cudaGetErrorEnum(err)); \
		cudaDeviceReset(); assert(0); \
    }
}

__global__ void copy_to_fft_input(volatile float * __restrict__ fft_input, 
                                  const float * w_coefficients_device,
                                  const int n_fft_coeffs,
                                  const int n_fft_coeffs_half,
                                  const int n_terms)
{
    register int i, j;
    register int TID = threadIdx.x + blockIdx.x * blockDim.x;
    if (TID >= n_terms * n_fft_coeffs_half * n_fft_coeffs_half)
        return;

    register int current_term = TID / (n_fft_coeffs_half * n_fft_coeffs_half);
    register int current_loc = TID % (n_fft_coeffs_half * n_fft_coeffs_half);
    
    i = current_loc / n_fft_coeffs_half;
    j = current_loc % n_fft_coeffs_half;

    fft_input[current_term * (n_fft_coeffs * n_fft_coeffs) + i * n_fft_coeffs + j] = w_coefficients_device[current_term + current_loc * n_terms];
}

__global__ void copy_from_fft_output(volatile float * __restrict__ y_tilde_values, 
    const float * fft_output,
    const int n_fft_coeffs,
    const int n_fft_coeffs_half,
    const int n_terms)
{
    register int i, j;
    register int TID = threadIdx.x + blockIdx.x * blockDim.x;
    if (TID >= n_terms * n_fft_coeffs_half * n_fft_coeffs_half)
        return;

    register int current_term = TID / (n_fft_coeffs_half * n_fft_coeffs_half);
    register int current_loc = TID % (n_fft_coeffs_half * n_fft_coeffs_half);

    i = current_loc / n_fft_coeffs_half + n_fft_coeffs_half;
    j = current_loc % n_fft_coeffs_half + n_fft_coeffs_half;

    y_tilde_values[current_term + n_terms * current_loc] = fft_output[current_term * (n_fft_coeffs * n_fft_coeffs) + i * n_fft_coeffs + j] / (float) (n_fft_coeffs * n_fft_coeffs);
}

__global__ void compute_point_box_idx(volatile int * __restrict__ point_box_idx,
                                      volatile float * __restrict__ x_in_box,
                                      volatile float * __restrict__ y_in_box,
                                      const float * const xs,
                                      const float * const ys,
                                      const float * const box_lower_bounds,
                                      const float coord_min,
                                      const float box_width,
                                      const int n_boxes,
                                      const int n_total_boxes,
                                      const int N) 
{
    register int TID = threadIdx.x + blockIdx.x * blockDim.x;
    if (TID >= N)
        return;
    
    register int x_idx = (int) ((xs[TID] - coord_min) / box_width);
    register int y_idx = (int) ((ys[TID] - coord_min) / box_width);

    x_idx = max(0, x_idx);
    x_idx = min(n_boxes - 1, x_idx);

    y_idx = max(0, y_idx);
    y_idx = min(n_boxes - 1, y_idx);

    register int box_idx = y_idx * n_boxes + x_idx;
    point_box_idx[TID] = box_idx;

    x_in_box[TID] = (xs[TID] - box_lower_bounds[box_idx]) / box_width;
    y_in_box[TID] = (ys[TID] - box_lower_bounds[n_total_boxes + box_idx]) / box_width;
}

void precompute_2d(float x_max, float x_min, float y_max, float y_min, int n_boxes, int n_interpolation_points,
                   kernel_type_2d kernel, float *box_lower_bounds, float *box_upper_bounds, float *y_tilde_spacings,
                   float *y_tilde, float *x_tilde, complex<float> *fft_kernel_tilde) {
    /*
     * Set up the boxes
     */
    int n_total_boxes = n_boxes * n_boxes;
    float box_width = (x_max - x_min) / (float) n_boxes;

    // Left and right bounds of each box, first the lower bounds in the x direction, then in the y direction
    for (int i = 0; i < n_boxes; i++) {
        for (int j = 0; j < n_boxes; j++) {
            box_lower_bounds[i * n_boxes + j] = j * box_width + x_min;
            box_upper_bounds[i * n_boxes + j] = (j + 1) * box_width + x_min;

            box_lower_bounds[n_total_boxes + i * n_boxes + j] = i * box_width + y_min;
            box_upper_bounds[n_total_boxes + i * n_boxes + j] = (i + 1) * box_width + y_min;
        }
    }

    // Coordinates of each (equispaced) interpolation node for a single box
    float h = 1 / (float) n_interpolation_points;
    y_tilde_spacings[0] = h / 2;
    for (int i = 1; i < n_interpolation_points; i++) {
        y_tilde_spacings[i] = y_tilde_spacings[i - 1] + h;
    }

    // Coordinates of all the equispaced interpolation points
    int n_interpolation_points_1d = n_interpolation_points * n_boxes;
    int n_fft_coeffs = 2 * n_interpolation_points_1d;

    h = h * box_width;
    x_tilde[0] = x_min + h / 2;
    y_tilde[0] = y_min + h / 2;
    for (int i = 1; i < n_interpolation_points_1d; i++) {
        x_tilde[i] = x_tilde[i - 1] + h;
        y_tilde[i] = y_tilde[i - 1] + h;
    }

    /*
     * Evaluate the kernel at the interpolation nodes and form the embedded generating kernel vector for a circulant
     * matrix
     */
    auto *kernel_tilde = new float[n_fft_coeffs * n_fft_coeffs]();
    for (int i = 0; i < n_interpolation_points_1d; i++) {
        for (int j = 0; j < n_interpolation_points_1d; j++) {
            float tmp = kernel(y_tilde[0], x_tilde[0], y_tilde[i], x_tilde[j]);
            kernel_tilde[(n_interpolation_points_1d + i) * n_fft_coeffs + (n_interpolation_points_1d + j)] = tmp;
            kernel_tilde[(n_interpolation_points_1d - i) * n_fft_coeffs + (n_interpolation_points_1d + j)] = tmp;
            kernel_tilde[(n_interpolation_points_1d + i) * n_fft_coeffs + (n_interpolation_points_1d - j)] = tmp;
            kernel_tilde[(n_interpolation_points_1d - i) * n_fft_coeffs + (n_interpolation_points_1d - j)] = tmp;
        }
    }

    // Precompute the FFT of the kernel generating matrix
    fftwf_plan p = fftwf_plan_dft_r2c_2d(n_fft_coeffs, n_fft_coeffs, kernel_tilde,
                                       reinterpret_cast<fftwf_complex *>(fft_kernel_tilde), FFTW_ESTIMATE);
    fftwf_execute(p);

    fftwf_destroy_plan(p);
    delete[] kernel_tilde;
}


void n_body_fft_2d(int N, int n_terms, float *xs, float *ys, float *chargesQij, int n_boxes,
                   int n_interpolation_points, float *box_lower_bounds, float *box_upper_bounds,
                   float *y_tilde_spacings, complex<float> *fft_kernel_tilde, float *potentialQij, unsigned int nthreads) {
    // std::cout << "start" << std::endl;
    const int num_threads = 1024;
    int num_blocks = (N + num_threads - 1) / num_threads;
    int n_total_boxes = n_boxes * n_boxes;
    int total_interpolation_points = n_total_boxes * n_interpolation_points * n_interpolation_points;

    float coord_min = box_lower_bounds[0];
    float box_width = box_upper_bounds[0] - box_lower_bounds[0];

    // auto *point_box_idx = new int[N];
    thrust::device_vector<int> point_box_idx_device(N);
    thrust::device_vector<float> x_in_box_device(N);
    thrust::device_vector<float> y_in_box_device(N);
    thrust::device_vector<float> xs_device(xs, xs + N);
    thrust::device_vector<float> ys_device(ys, ys + N);
    thrust::device_vector<float> box_lower_bounds_device(box_lower_bounds, box_lower_bounds + 2 * n_total_boxes);
    compute_point_box_idx<<<num_blocks, num_threads>>>(
        thrust::raw_pointer_cast(point_box_idx_device.data()),
        thrust::raw_pointer_cast(x_in_box_device.data()),
        thrust::raw_pointer_cast(y_in_box_device.data()),
        thrust::raw_pointer_cast(xs_device.data()), 
        thrust::raw_pointer_cast(ys_device.data()),
        thrust::raw_pointer_cast(box_lower_bounds_device.data()),
        coord_min,
        box_width,
        n_boxes,
        n_total_boxes,
        N
    );

    GpuErrorCheck(cudaDeviceSynchronize());
    thrust::host_vector<int> point_box_idx(point_box_idx_device.begin(), point_box_idx_device.end());
    thrust::host_vector<float> x_in_box(x_in_box_device.begin(), x_in_box_device.end());
    thrust::host_vector<float> y_in_box(y_in_box_device.begin(), y_in_box_device.end());

    // Compute the relative position of each point in its box in the interval [0, 1]
    // auto *x_in_box = new float[N];
    // auto *y_in_box = new float[N];
    // for (int i = 0; i < N; i++) {
    //     int box_idx = point_box_idx[i];
    //     float x_min = box_lower_bounds[box_idx];
    //     float y_min = box_lower_bounds[n_total_boxes + box_idx];
    //     x_in_box[i] = (xs[i] - x_min) / box_width;
    //     y_in_box[i] = (ys[i] - y_min) / box_width;
    // }

    INITIALIZE_TIME
    START_TIME

    /*
     * Step 1: Interpolate kernel using Lagrange polynomials and compute the w coefficients
     */
    // Compute the interpolated values at each real point with each Lagrange polynomial in the `x` direction
    auto *x_interpolated_values = new float[N * n_interpolation_points];
    interpolate(n_interpolation_points, N, thrust::raw_pointer_cast(x_in_box.data()), y_tilde_spacings, x_interpolated_values);
    // Compute the interpolated values at each real point with each Lagrange polynomial in the `y` direction
    auto *y_interpolated_values = new float[N * n_interpolation_points];
    interpolate(n_interpolation_points, N, thrust::raw_pointer_cast(y_in_box.data()), y_tilde_spacings, y_interpolated_values);

    auto *w_coefficients = new float[total_interpolation_points * n_terms]();
    for (int i = 0; i < N; i++) {
        int box_idx = point_box_idx[i];
        int box_j = box_idx / n_boxes;
        int box_i = box_idx % n_boxes;
        for (int interp_i = 0; interp_i < n_interpolation_points; interp_i++) {
            for (int interp_j = 0; interp_j < n_interpolation_points; interp_j++) {
                // Compute the index of the point in the interpolation grid of points
                int idx = (box_i * n_interpolation_points + interp_i) * (n_boxes * n_interpolation_points) +
                          (box_j * n_interpolation_points) + interp_j;
                for (int d = 0; d < n_terms; d++) {
                    w_coefficients[idx * n_terms + d] +=
                            y_interpolated_values[interp_j * N + i] *
                            x_interpolated_values[interp_i * N + i] *
                            chargesQij[i * n_terms + d];
                }
            }
        }
    }

        END_TIME("Step 1");
        START_TIME;
    /*
     * Step 2: Compute the values v_{m, n} at the equispaced nodes, multiply the kernel matrix with the coefficients w
     */
    int n_fft_coeffs_half = n_interpolation_points * n_boxes;
    int n_fft_coeffs = 2 * n_interpolation_points * n_boxes;
    
    thrust::device_vector<float> w_coefficients_device(w_coefficients, w_coefficients + (total_interpolation_points * n_terms ));
    thrust::device_vector<float> y_tilde_values(total_interpolation_points * n_terms);

    // FFT of fft_input
    thrust::host_vector<float> y_tilde_values_host(total_interpolation_points * n_terms);

    // CUDA vectors
    thrust::device_vector<float> fft_input(n_terms * n_fft_coeffs * n_fft_coeffs);
    thrust::device_vector<thrust::complex<float>> fft_w_coefficients(n_terms * n_fft_coeffs * (n_fft_coeffs / 2 + 1));
    // std::cout << "starting copy" << std::endl;
    thrust::device_vector<thrust::complex<float>> fft_kernel_tilde_device((
        thrust::complex<float> *) fft_kernel_tilde, ((thrust::complex<float> *) fft_kernel_tilde) + n_fft_coeffs * (n_fft_coeffs / 2 + 1));
    
    thrust::device_vector<float> fft_output(n_terms * n_fft_coeffs * n_fft_coeffs);

    // New CUFFT plans
    cufftHandle plan_dft, plan_idft;
    cufftSafeCall(cufftCreate(&plan_dft));
    cufftSafeCall(cufftCreate(&plan_idft));

    size_t work_size_dft, work_size_idft;
    int n[2] = {n_fft_coeffs, n_fft_coeffs};
    cufftSafeCall(cufftMakePlanMany(plan_dft, 2, n, 
                                    NULL, 1, n_fft_coeffs * n_fft_coeffs,
                                    NULL, 1, n_fft_coeffs * (n_fft_coeffs / 2 + 1),
                                    CUFFT_R2C, n_terms, &work_size_dft));
    cufftSafeCall(cufftMakePlanMany(plan_idft, 2, n, 
                                    NULL, 1, n_fft_coeffs * (n_fft_coeffs / 2 + 1),
                                    NULL, 1, n_fft_coeffs * n_fft_coeffs,
                                    CUFFT_C2R, n_terms, &work_size_idft));
    
    num_blocks = ((n_terms * n_fft_coeffs_half * n_fft_coeffs_half) + num_threads - 1) / num_threads;
    copy_to_fft_input<<<num_blocks, num_threads>>>(
        thrust::raw_pointer_cast(fft_input.data()),
        thrust::raw_pointer_cast(w_coefficients_device.data()),
        n_fft_coeffs,
        n_fft_coeffs_half,
        n_terms
    );
    GpuErrorCheck(cudaDeviceSynchronize());

    // Compute fft values at interpolated nodes
    cufftExecR2C(plan_dft, 
        reinterpret_cast<cufftReal *>(thrust::raw_pointer_cast(fft_input.data())), 
        reinterpret_cast<cufftComplex *>(thrust::raw_pointer_cast(fft_w_coefficients.data())));
    GpuErrorCheck(cudaDeviceSynchronize());

    // Take the broadcasted Hadamard product of a complex matrix and a complex vector
    tsnecuda::util::BroadcastMatrixVector(
        fft_w_coefficients, fft_kernel_tilde_device, n_fft_coeffs * (n_fft_coeffs / 2 + 1), n_terms, 
        thrust::multiplies<thrust::complex<float>>(), 0, thrust::complex<float>(1.0));

    

    // Invert the computed values at the interpolated nodes
    cufftExecC2R(plan_idft, 
        reinterpret_cast<cufftComplex *>(thrust::raw_pointer_cast(fft_w_coefficients.data())), 
        reinterpret_cast<cufftReal *>(thrust::raw_pointer_cast(fft_output.data())));
    GpuErrorCheck(cudaDeviceSynchronize());

    copy_from_fft_output<<<num_blocks, num_threads>>>(
        thrust::raw_pointer_cast(y_tilde_values.data()),
        thrust::raw_pointer_cast(fft_output.data()),
        n_fft_coeffs,
        n_fft_coeffs_half,
        n_terms
    );
    GpuErrorCheck(cudaDeviceSynchronize());

    thrust::copy(y_tilde_values.begin(), y_tilde_values.end(), y_tilde_values_host.begin());

    cufftSafeCall(cufftDestroy(plan_dft));
    cufftSafeCall(cufftDestroy(plan_idft));
    
    END_TIME("FFT");
    START_TIME
    /*
     * Step 3: Compute the potentials \tilde{\phi}
     */
    PARALLEL_FOR(nthreads,N, {
        int box_idx = point_box_idx[loop_i];
        int box_i = box_idx % n_boxes;
        int box_j = box_idx / n_boxes;
        for (int interp_i = 0; interp_i < n_interpolation_points; interp_i++) {
            for (int interp_j = 0; interp_j < n_interpolation_points; interp_j++) {
                for (int d = 0; d < n_terms; d++) {
                    // Compute the index of the point in the interpolation grid of points
                    int idx = (box_i * n_interpolation_points + interp_i) * (n_boxes * n_interpolation_points) +
                              (box_j * n_interpolation_points) + interp_j;
                    potentialQij[loop_i * n_terms + d] +=
                            x_interpolated_values[interp_i * N + loop_i] *
                            y_interpolated_values[interp_j * N + loop_i] *
                            y_tilde_values_host[idx * n_terms + d];
                }
            }
        }
    });
    END_TIME("Step 3");
    // delete[] point_box_idx;
    delete[] x_interpolated_values;
    delete[] y_interpolated_values;
    delete[] w_coefficients;
    // delete[] y_tilde_values;
    // delete[] x_in_box;
    // delete[] y_in_box;

    // std::cout << "done" << std::endl;
}


void precompute(float y_min, float y_max, int n_boxes, int n_interpolation_points, kernel_type kernel,
                float *box_lower_bounds, float *box_upper_bounds, float *y_tilde_spacing, float *y_tilde,
                complex<float> *fft_kernel_vector) {
    /*
     * Set up the boxes
     */
    float box_width = (y_max - y_min) / (float) n_boxes;
    // Compute the left and right bounds of each box
    for (int box_idx = 0; box_idx < n_boxes; box_idx++) {
        box_lower_bounds[box_idx] = box_idx * box_width + y_min;
        box_upper_bounds[box_idx] = (box_idx + 1) * box_width + y_min;
    }

    int total_interpolation_points = n_interpolation_points * n_boxes;
    // Coordinates of each equispaced interpolation point for a single box. This equally spaces them between [0, 1]
    // with equal space between the points and half that space between the boundary point and the closest boundary point
    // e.g. [0.1, 0.3, 0.5, 0.7, 0.9] with spacings [0.1, 0.2, 0.2, 0.2, 0.2, 0.1], respectively. This ensures that the
    // nodes will still be equispaced across box boundaries
    float h = 1 / (float) n_interpolation_points;
    y_tilde_spacing[0] = h / 2;
    for (int i = 1; i < n_interpolation_points; i++) {
        y_tilde_spacing[i] = y_tilde_spacing[i - 1] + h;
    }

    // Coordinates of all the equispaced interpolation points
    h = h * box_width;
    y_tilde[0] = y_min + h / 2;
    for (int i = 1; i < total_interpolation_points; i++) {
        y_tilde[i] = y_tilde[i - 1] + h;
    }

    /*
     * Evaluate the kernel at the interpolation nodes and form the embedded generating kernel vector for a circulant
     * matrix
     */
    auto *kernel_vector = new complex<float>[2 * total_interpolation_points]();
    // Compute the generating vector x between points K(y_i, y_j) where i = 0, j = 0:N-1
    // [0 0 0 0 0 5 4 3 2 1] for linear kernel
    // This evaluates the Cauchy kernel centered on y_tilde[0] to all the other points
    for (int i = 0; i < total_interpolation_points; i++) {
        kernel_vector[total_interpolation_points + i].real(kernel(y_tilde[0], y_tilde[i]));
    }
    // This part symmetrizes the vector, this embeds the Toeplitz generating vector into the circulant generating vector
    // but also has the nice property of symmetrizing the Cauchy kernel, which is probably planned
    // [0 1 2 3 4 5 4 3 2 1] for linear kernel
    for (int i = 1; i < total_interpolation_points; i++) {
        kernel_vector[i].real(kernel_vector[2 * total_interpolation_points - i].real());
    }

    // Precompute the FFT of the kernel generating vector
    fftwf_plan p = fftwf_plan_dft_1d(2 * total_interpolation_points, reinterpret_cast<fftwf_complex *>(kernel_vector),
                                   reinterpret_cast<fftwf_complex *>(fft_kernel_vector), FFTW_FORWARD, FFTW_ESTIMATE);
    fftwf_execute(p);
    fftwf_destroy_plan(p);

    delete[] kernel_vector;
}


void interpolate(int n_interpolation_points, int N, const float *y_in_box, const float *y_tilde_spacings,
                 float *interpolated_values) {
    // The denominators are the same across the interpolants, so we only need to compute them once
    auto *denominator = new float[n_interpolation_points];
    for (int i = 0; i < n_interpolation_points; i++) {
        denominator[i] = 1;
        for (int j = 0; j < n_interpolation_points; j++) {
            if (i != j) {
                denominator[i] *= y_tilde_spacings[i] - y_tilde_spacings[j];
            }
        }
    }
    // Compute the numerators and the interpolant value
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < n_interpolation_points; j++) {
            interpolated_values[j * N + i] = 1;
            for (int k = 0; k < n_interpolation_points; k++) {
                if (j != k) {
                    interpolated_values[j * N + i] *= y_in_box[i] - y_tilde_spacings[k];
                }
            }
            interpolated_values[j * N + i] /= denominator[j];
        }
    }

    delete[] denominator;
}


void nbodyfft(int N, int n_terms, float *Y, float *chargesQij, int n_boxes, int n_interpolation_points,
              float *box_lower_bounds, float *box_upper_bounds, float *y_tilde_spacings, float *y_tilde,
              complex<float> *fft_kernel_vector, float *potentialsQij) {
    int total_interpolation_points = n_interpolation_points * n_boxes;

    float coord_min = box_lower_bounds[0];
    float box_width = box_upper_bounds[0] - box_lower_bounds[0];

    // Determine which box each point belongs to
    auto *point_box_idx = new int[N];
    for (int i = 0; i < N; i++) {
        auto box_idx = static_cast<int>((Y[i] - coord_min) / box_width);
        // The right most point maps directly into `n_boxes`, while it should belong to the last box
        if (box_idx >= n_boxes) {
            box_idx = n_boxes - 1;
        }
        point_box_idx[i] = box_idx;
    }

    // Compute the relative position of each point in its box in the interval [0, 1]
    auto *y_in_box = new float[N];
    for (int i = 0; i < N; i++) {
        int box_idx = point_box_idx[i];
        float box_min = box_lower_bounds[box_idx];
        y_in_box[i] = (Y[i] - box_min) / box_width;
    }

    /*
     * Step 1: Interpolate kernel using Lagrange polynomials and compute the w coefficients
     */
    // Compute the interpolated values at each real point with each Lagrange polynomial
    auto *interpolated_values = new float[n_interpolation_points * N];
    interpolate(n_interpolation_points, N, y_in_box, y_tilde_spacings, interpolated_values);

    auto *w_coefficients = new float[total_interpolation_points * n_terms]();
    for (int i = 0; i < N; i++) {
        int box_idx = point_box_idx[i] * n_interpolation_points;
        for (int interp_idx = 0; interp_idx < n_interpolation_points; interp_idx++) {
            for (int d = 0; d < n_terms; d++) {
                w_coefficients[(box_idx + interp_idx) * n_terms + d] +=
                        interpolated_values[interp_idx * N + i] * chargesQij[i * n_terms + d];
            }
        }
    }

    // `embedded_w_coefficients` is just a vector of zeros prepended to `w_coefficients`, this (probably) matches the
    // dimensions of the kernel matrix K and since we embedded the generating vector by prepending values, we have to do
    // the same here
    auto *embedded_w_coefficients = new float[2 * total_interpolation_points * n_terms]();
    for (int i = 0; i < total_interpolation_points; i++) {
        for (int d = 0; d < n_terms; d++) {
            embedded_w_coefficients[(total_interpolation_points + i) * n_terms + d] = w_coefficients[i * n_terms + d];
        }
    }

    /*
     * Step 2: Compute the values v_{m, n} at the equispaced nodes, multiply the kernel matrix with the coefficients w
     */
    auto *fft_w_coefficients = new complex<float>[2 * total_interpolation_points];
    auto *y_tilde_values = new float[total_interpolation_points * n_terms]();

    fftwf_plan plan_dft, plan_idft;
    plan_dft = fftwf_plan_dft_1d(2 * total_interpolation_points, reinterpret_cast<fftwf_complex *>(fft_w_coefficients),
                                reinterpret_cast<fftwf_complex *>(fft_w_coefficients), FFTW_FORWARD, FFTW_ESTIMATE);
    plan_idft = fftwf_plan_dft_1d(2 * total_interpolation_points, reinterpret_cast<fftwf_complex *>(fft_w_coefficients),
                                 reinterpret_cast<fftwf_complex *>(fft_w_coefficients), FFTW_BACKWARD, FFTW_ESTIMATE);

    for (int d = 0; d < n_terms; d++) {
        for (int i = 0; i < 2 * total_interpolation_points; i++) {
            fft_w_coefficients[i].real(embedded_w_coefficients[i * n_terms + d]);
        }
        fftwf_execute(plan_dft);

        // Take the Hadamard product of two complex vectors
        for (int i = 0; i < 2 * total_interpolation_points; i++) {
            float x_ = fft_w_coefficients[i].real();
            float y_ = fft_w_coefficients[i].imag();
            float u_ = fft_kernel_vector[i].real();
            float v_ = fft_kernel_vector[i].imag();
            fft_w_coefficients[i].real(x_ * u_ - y_ * v_);
            fft_w_coefficients[i].imag(x_ * v_ + y_ * u_);
        }

        // Invert the computed values at the interpolated nodes, unfortunate naming but it's better to do IDFT inplace
        fftwf_execute(plan_idft);

        for (int i = 0; i < total_interpolation_points; i++) {
            // FFTW doesn't perform IDFT normalization, so we have to do it ourselves. This is done by multiplying the
            // result with the number of points in the input
            y_tilde_values[i * n_terms + d] = fft_w_coefficients[i].real() / (total_interpolation_points * 2.0);
        }
    }

    fftwf_destroy_plan(plan_dft);
    fftwf_destroy_plan(plan_idft);
    delete[] fft_w_coefficients;

    /*
     * Step 3: Compute the potentials \tilde{\phi}
     */
    for (int i = 0; i < N; i++) {
        int box_idx = point_box_idx[i] * n_interpolation_points;
        for (int j = 0; j < n_interpolation_points; j++) {
            for (int d = 0; d < n_terms; d++) {
                potentialsQij[i * n_terms + d] +=
                        interpolated_values[j * N + i] * y_tilde_values[(box_idx + j) * n_terms + d];
            }
        }
    }

    delete[] point_box_idx;
    delete[] y_in_box;
    delete[] interpolated_values;
    delete[] w_coefficients;
    delete[] y_tilde_values;
    delete[] embedded_w_coefficients;
}
