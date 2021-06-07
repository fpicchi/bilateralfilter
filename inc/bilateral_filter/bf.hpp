/**
 * @file bf.hpp
 * @author Mirco De Marchi, Federico Picchi
 * @date June 2021
 * 
 * @copyright Copyright (c) 2021
 * 
 */


#include <opencv2/opencv.hpp>
#include <vector>


cv::Mat bf_sequential(const cv::Mat &source, 
    int diameter, double sigma_i, double sigma_s);

cv::Mat bf_sequential_cv(const cv::Mat &source, 
    int diameter, double sigma_i, double sigma_s);

cv::Mat bf_parallel(const cv::Mat &source, 
    int diameter, double sigma_i, double sigma_s);

cv::Mat bf_parallel_naive(const cv::Mat &source,
    int diameter, double sigma_i, double sigma_s);

cv::Mat bf_parallel_cv(const cv::Mat &source, 
    int diameter, double sigma_i, double sigma_s);