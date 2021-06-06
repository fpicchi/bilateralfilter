/**
 * @file bf_sequential.cpp
 * @author Mirco De Marchi
 * @brief CPU-based sequential implementation of Bilateral filter
 * @date June 2021
 * 
 * @copyright Copyright (c) 2021
 * 
 */

#include "bilateral_filter/bf.hpp"
#include <iostream>

void bf_sequential_apply(const cv::Mat &temp, const cv::Mat &dst, int radius,
    int i, int j, double *gi, double *gs, int *space_coord, int maxk)
{
    const uchar *sptr = temp.data + (i + radius) * temp.step + radius;
    uchar *dptr = dst.data + i * dst.step;

    double sum = 0, wsum = 0;
    int val0 = sptr[j]; //< Center of the template.
    for (int k = 0; k < maxk; k++)
    {
        int val = sptr[j + space_coord[k]];
        // The weight is gaussian space * color space.
        double w = gs[k] * gi[abs(val - val0)]; 
        sum += val * w;
        wsum += w;
    }
    dptr[j] = (uchar)std::round(sum / wsum);
}

cv::Mat bf_sequential(const cv::Mat &src, int ksize, 
    double sigma_i, double sigma_s)
{
    cv::Mat dst = cv::Mat::zeros(src.rows, src.cols, CV_8U);
    int radius = ksize / 2;

    // Create an image with a border.
    cv::Mat temp;
    cv::copyMakeBorder(src, temp, radius, radius, radius, radius, 
        cv::BorderTypes::BORDER_REFLECT_101);

    // Init color weight.
    double coeff_i = -0.5 / (sigma_i * sigma_i);
    std::vector<double> gi_vec(256);
    double *gi = &gi_vec[0];
    for (int i = 0; i < 256; i++)
        gi[i] = exp(i * i * coeff_i);

    // Generate gaussian space.
    std::vector<double> gs_vec(ksize * ksize);
    std::vector<int> space_coord_vec(ksize * ksize); //< Save here coord.
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
    std::cout << "STEP: " << temp.step << "\n";
    for (int i = 0; i < src.rows; i++)
    {
        for (int j = 0; j < src.cols; j++)
        {
            bf_sequential_apply(temp, dst, radius, i, j, gi, gs, space_coord, 
                maxk);
        }
    }
    return dst;
}
