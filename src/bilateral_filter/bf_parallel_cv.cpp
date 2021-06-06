/**
 * @file bf_parallel_cv.cpp
 * @author your name (you@domain.com)
 * @brief OpenCV bilateral filter. Parallel.
 * @date June 2021
 * 
 * @copyright Copyright (c) 2021
 * 
 */

#include "bilateral_filter/bf.hpp"


cv::Mat bf_parallel_cv(const cv::Mat &source, 
    int diameter, double sigma_i, double sigma_s)
{
#if defined(OPENCV_CUDA)
    cv::Mat ret;
    cv::bilateralFilter(source, ret, diameter, sigma_i, sigma_s, 
        cv::BORDER_REFLECT_101);
    return ret;
#else
    return source;
#endif // defined(OPENCV_CUDA)
}