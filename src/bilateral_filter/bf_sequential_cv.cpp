/**
 * @file bf_sequential_cv.cpp
 * @author Mirco De Marchi
 * @brief OpenCV bilateral filter. Sequential.
 * @date June 2021
 * 
 * @copyright Copyright (c) 2021
 * 
 */

#include "bilateral_filter/bf.hpp"


cv::Mat bf_sequential_cv(const cv::Mat &source, 
    int diameter, double sigma_i, double sigma_s)
{
#if defined(OPENCV_CUDA)
    cv::Mat ret;
    cv::cuda::GpuMat frame_in_d, frame_out_d;
    frame_in_d.upload(source);
    cv::cuda::bilateralFilter(frame_in_d, frame_out_d, diameter, 
        sigma_i, sigma_s, cv::BORDER_REFLECT_101);
    frame_out_d.download(ret);
    return ret;
#else
    return source;
#endif // defined(OPENCV_CUDA)
}