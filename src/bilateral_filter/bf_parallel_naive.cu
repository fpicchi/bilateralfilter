/**
 * @file bf_parallel.cu
 * @author Federico Picchi
 * @brief CUDA-based implementation of Bilateral Filter
 * @date June 2021
 * 
 * @copyright Copyright (c) 2021
 * 
 */


#include "bilateral_filter/bf.hpp"
#if defined(_DEBUG)
#include <iostream>
#endif

inline void handleError(cudaError_t err, int line) {
#if defined(_DEBUG)
    if (err) {
        std::cerr << __FILE__ << ": ERROR " << err << " CALLING CUDA FUNCTION IN LINE: " << line << "\n";
        exit(1);
    }
#endif
    return;
}

__global__
void bf_parallel_naive_k(const uchar*  const source, 
                       uchar*  const destination, 
                 const int           radius,
                 const double* const gi,
                 const double* const gs,
                 const int*    const space_coord,
                 const int           maxk,
                 const size_t        width,
                 const size_t        height,
                 const size_t        s_step,
                 const size_t        d_step) {
    const int global_j = threadIdx.x + blockIdx.x * blockDim.x;
    const int global_i = threadIdx.y + blockIdx.y * blockDim.y;
    if (global_i >= height || global_j >= width) return;
    const uchar* const sptr = source + (global_i + radius) * s_step + radius;
    uchar* const dptr = destination + global_i * d_step;

    double sum = 0, wsum = 0;
    const int val0 = sptr[global_j]; //< Center of the template.
    for (int k = 0; k < maxk; k++)
    {
        const int val = sptr[global_j + space_coord[k]];
        // The weight is gaussian space * color space.
        const double w = gs[k] * gi[abs(val - val0)];
        sum += val * w;
        wsum += w;
    }
    dptr[global_j] = (uchar)lround(sum / wsum);
}

cv::Mat bf_parallel_naive(const cv::Mat &source, 
    const int diameter, const double sigma_i, const double sigma_s)
{
    cv::Mat dst = cv::Mat::zeros(source.rows, source.cols, CV_8U);
    int radius = diameter / 2;

    // Create an image with a border.
    cv::Mat temp;
    cv::copyMakeBorder(source, temp, radius, radius, radius, radius,
        cv::BorderTypes::BORDER_REFLECT_101);

    // Init color weight.
    double coeff_i = -0.5 / (sigma_i * sigma_i);
    std::vector<double> gi_vec(256);
    double *gi = &gi_vec[0];
    for (int i = 0; i < 256; i++)
        gi[i] = exp(i * i * coeff_i);

    // Generate gaussian space.
    std::vector<double> gs_vec(diameter * diameter);
    std::vector<int> space_coord_vec(diameter * diameter); //< Save here coord.
    double *gs = &gs_vec[0];
    int    *space_coord = &space_coord_vec[0];
    const double coeff_s = -0.5 / (sigma_s * sigma_s);
    int maxk = 0;
    for (int i = -radius; i <= radius; i++)
    {
        for (int j = -radius; j <= radius; j++)
        {
            double r = sqrt(i * i + j * j);
            if (r > radius) //< Circle.
                continue;
            gs[maxk] = exp(r * r * coeff_s);
            space_coord[maxk++] = i * (int)temp.step + j;
        }
    }

    // Copy data to device
    uchar* temp_d;
    handleError(cudaMalloc(&temp_d, temp.total()), __LINE__);
    handleError(cudaMemcpy(temp_d, temp.data, temp.total(), cudaMemcpyHostToDevice), __LINE__);
    uchar* dst_d;
    handleError(cudaMalloc(&dst_d, dst.total()), __LINE__);
    handleError(cudaMemcpy(dst_d, dst.data, dst.total(), cudaMemcpyHostToDevice), __LINE__);
    double* gs_d;
    handleError(cudaMalloc(&gs_d, diameter * diameter * sizeof(double)), __LINE__);
    handleError(cudaMemcpy(gs_d, gs, diameter * diameter * sizeof(double), cudaMemcpyHostToDevice), __LINE__);
    double* gi_d;
    handleError(cudaMalloc(&gi_d, 256 * sizeof(double)), __LINE__);
    handleError(cudaMemcpy(gi_d, gi, 256 * sizeof(double), cudaMemcpyHostToDevice), __LINE__);
    int* space_coord_d;
    handleError(cudaMalloc(&space_coord_d, diameter * diameter * sizeof(int)), __LINE__);
    handleError(cudaMemcpy(space_coord_d, space_coord, diameter * diameter * sizeof(int), cudaMemcpyHostToDevice), __LINE__);
    //Calculate optimal CUDA configuration
    int blockSize;
    int minGridSize;
    cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize,
                                        bf_parallel_naive_k, 0, 0);
    // Round up according to matrix size 
    blockSize = (int)sqrt(blockSize);
    dim3 gridSize_2D((source.cols + blockSize - 1) / blockSize, (source.rows + blockSize - 1) / blockSize);
    dim3 blockSize_2D(blockSize, blockSize);
    // Filtering process
    bf_parallel_naive_k << < gridSize_2D, blockSize_2D >> > (temp_d, dst_d, radius, gi_d, gs_d,
                                                space_coord_d, maxk, source.cols,
                                                source.rows, temp.step, dst.step);

    handleError(cudaDeviceSynchronize(), __LINE__);
    // Copy data from device
    handleError(cudaMemcpy(dst.data, dst_d, dst.total(), cudaMemcpyDeviceToHost), __LINE__);
    handleError(cudaFree(temp_d), __LINE__);
    handleError(cudaFree(dst_d), __LINE__);
    handleError(cudaFree(gs_d), __LINE__);
    handleError(cudaFree(gi_d), __LINE__);
    handleError(cudaFree(space_coord_d), __LINE__);

    return dst;
}