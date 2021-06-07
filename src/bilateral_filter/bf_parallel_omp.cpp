/**
 * @file bf_sequential.cpp
 * @author Mirco De Marchi, Federico Picchi
 * @brief CPU-based parallel implementation of Bilateral filter
 * @date June 2021
 * 
 * @copyright Copyright (c) 2021
 * 
 */

#include "bilateral_filter/bf.hpp"

extern void bf_sequential_apply(const cv::Mat &temp, const cv::Mat &dst, int radius,
    int i, int j, double *gi, double *gs, int *space_coord, int maxk);

cv::Mat bf_parallel_omp(const cv::Mat &source, int diameter,
    double sigma_i, double sigma_s)
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
    std::vector<int> space_coord_vec(diameter * diameter);
    double *gs = &gs_vec[0];
    int    *space_coord = &space_coord_vec[0];
    double coeff_s = -0.5 / (sigma_s * sigma_s);
    int maxk = 0;
    for (int i = -radius; i <= radius; i++)
    {
        for (int j = -radius; j <= radius; j++)
        {
            double r = sqrt(i * i + j * j);
            if (r > radius) //< Circle.
                continue;
            gs[maxk] = exp(r * r * coeff_s); 
            space_coord[maxk++] = i * temp.step + j;
        }
    }

    // Filtering process
    #pragma omp parallel for
    for (int i = 0; i < source.rows; i++)
    {
        for (int j = 0; j < source.cols; j++)
        {
            bf_sequential_apply(temp, dst, radius, i, j, gi, gs, space_coord, 
                maxk);
        }
    }
    return dst;
}
