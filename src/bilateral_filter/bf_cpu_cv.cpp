/**
 * @file bf_cpu_cv.cpp
 * @author Mirco De Marchi
 * @brief CPU-based OpenCV bilateral filter.
 * @date June 2021
 * 
 * @copyright Copyright (c) 2021
 * 
 */

#include "bilateral_filter/bf.hpp"

cv::Mat bf_cpu_cv(const cv::Mat &source, 
    int diameter, double sigma_i, double sigma_s)
{
    cv::Mat ret;
    cv::bilateralFilter(source, ret, diameter, sigma_i, sigma_s,
        cv::BORDER_REFLECT_101);
    return ret;
}