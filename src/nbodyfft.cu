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

__global__ void interpolate_device(
    volatile float * __restrict__ interpolated_values,
    const float * const y_in_box,
    const float * const y_tilde_spacings,
    const float * const denominator,
    const int n_interpolation_points,
    const int N)
{
    register int TID, i, j, k;
    register float value, ybox_i;

    TID = threadIdx.x + blockIdx.x * blockDim.x;
    if (TID >= N * n_interpolation_points)
        return;

    i = TID % N;
    j = TID / N;
    
    value = 1;
    ybox_i = y_in_box[i];

    for (k = 0; k < n_interpolation_points; k++) {
        if (j != k) {
            value *= ybox_i - y_tilde_spacings[k];
        }
    }

    interpolated_values[j * N + i] = value / denominator[j];
}

// for (int i = 0; i < N; i++) {
//     for (int j = 0; j < n_interpolation_points; j++) {
//         interpolated_values[j * N + i] = 1;
//         for (int k = 0; k < n_interpolation_points; k++) {
//             if (j != k) {
//                 interpolated_values[j * N + i] *= y_in_box[i] - y_tilde_spacings[k];
//             }
//         }
//         interpolated_values[j * N + i] /= denominator[j];
//     }
// }


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
                   float *y_tilde_spacings, complex<float> *fft_kernel_tilde, float *potentialQij, const float *denominator, unsigned int nthreads) {
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
    thrust::device_vector<float> y_tilde_spacings_device(y_tilde_spacings, y_tilde_spacings + n_interpolation_points);
    thrust::device_vector<float> denominator_device(denominator, denominator + n_interpolation_points);

     // Compute box indices and the relative position of each point in its box in the interval [0, 1]
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

    INITIALIZE_TIME
    START_TIME

    /*
     * Step 1: Interpolate kernel using Lagrange polynomials and compute the w coefficients
     */
    // Compute the interpolated values at each real point with each Lagrange polynomial in the `x` direction
    
    thrust::device_vector<float> x_interpolated_values_device(N * n_interpolation_points);
    thrust::device_vector<float> y_interpolated_values_device(N * n_interpolation_points);

    num_blocks = (N * n_interpolation_points + num_threads - 1) / num_threads;
    interpolate_device<<<num_blocks, num_threads>>>(
        thrust::raw_pointer_cast(x_interpolated_values_device.data()),
        thrust::raw_pointer_cast(x_in_box_device.data()),
        thrust::raw_pointer_cast(y_tilde_spacings_device.data()),
        thrust::raw_pointer_cast(denominator_device.data()),
        n_interpolation_points,
        N
    );
    GpuErrorCheck(cudaDeviceSynchronize());

    interpolate_device<<<num_blocks, num_threads>>>(
        thrust::raw_pointer_cast(y_interpolated_values_device.data()),
        thrust::raw_pointer_cast(y_in_box_device.data()),
        thrust::raw_pointer_cast(y_tilde_spacings_device.data()),
        thrust::raw_pointer_cast(denominator_device.data()),
        n_interpolation_points,
        N
    );
    GpuErrorCheck(cudaDeviceSynchronize());

    thrust::host_vector<float> x_interpolated_values(x_interpolated_values_device.begin(), x_interpolated_values_device.end());
    thrust::host_vector<float> y_interpolated_values(y_interpolated_values_device.begin(), y_interpolated_values_device.end());
    // auto *x_interpolated_values = new float[N * n_interpolation_points];
    // interpolate(n_interpolation_points, N, thrust::raw_pointer_cast(x_in_box.data()), y_tilde_spacings, x_interpolated_values, denominator);
    // // Compute the interpolated values at each real point with each Lagrange polynomial in the `y` direction
    // auto *y_interpolated_values = new float[N * n_interpolation_points];
    // interpolate(n_interpolation_points, N, thrust::raw_pointer_cast(y_in_box.data()), y_tilde_spacings, y_interpolated_values, denominator);

    auto *w_coefficients = new float[total_interpolation_points * n_terms]();
    thrust::device_vecor<float> w_coefficients_device(total_interpolation_points * n_terms);
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
    
    // thrust::device_vector<float> w_coefficients_device(w_coefficients, w_coefficients + (total_interpolation_points * n_terms ));
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
    // delete[] x_interpolated_values;
    // delete[] y_interpolated_values;
    delete[] w_coefficients;
    // delete[] y_tilde_values;
    // delete[] x_in_box;
    // delete[] y_in_box;

    // std::cout << "done" << std::endl;
}


void interpolate(int n_interpolation_points, int N, const float *y_in_box, const float *y_tilde_spacings,
                 float *interpolated_values, const float *denominator) {
    // The denominators are the same across the interpolants, so we only need to compute them once
    // auto *denominator = new float[n_interpolation_points];
    // for (int i = 0; i < n_interpolation_points; i++) {
    //     denominator[i] = 1;
    //     for (int j = 0; j < n_interpolation_points; j++) {
    //         if (i != j) {
    //             denominator[i] *= y_tilde_spacings[i] - y_tilde_spacings[j];
    //         }
    //     }
    // }

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

    // delete[] denominator;
}