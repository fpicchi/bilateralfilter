/**
 * @file main.cpp
 * @author Mirco De Marchi, Federico Picchi
 * @brief Bilateral Filter benchmark
 * @version 0.1
 * @date June 2021
 * 
 * @copyright Copyright (c) 2021
 * 
 */

#include "bilateral_filter/bf.hpp"

#include <functional>
#include <experimental/filesystem>
#include <chrono>
#include <vector>
#include <iostream>
#include <string>
#include <numeric>

#define CHECK_PARALLEL 1
#define SPEEDUP_PARALLEL 1

namespace fs = std::experimental::filesystem;

static const int    param_diameter = 9;
static const double param_sigma_i  = 70.0;
static const double param_sigma_s  = 70.0;

static const fs::path data_dp = fs::path(__FILE__).parent_path() / ".." 
    / "data";
static const fs::path input_dp = data_dp / "input";
static const fs::path output_dp = data_dp / "output";

static const fs::path output_bf_parallel_dp = output_dp 
    / "bf_parallel";
static const fs::path output_bf_parallel_cv_dp = output_dp 
    / "bf_parallel_cv";
static const fs::path output_bf_sequential_dp = output_dp 
    / "bf_sequential";
static const fs::path output_bf_sequential_cv_dp = output_dp 
    / "bf_sequential_cv";

static cv::Mat test(const std::string info, const fs::path &fp,
    const cv::Mat &input_mat,
    std::function<cv::Mat(const cv::Mat&, int, double, double)> cb,
    std::vector<float> &times)
{
    std::cout << "[" << fp.filename() << "]" << info << std::endl;
    auto start = std::chrono::system_clock::now();
    auto ret = cb(input_mat, param_diameter, 
        param_sigma_i, param_sigma_s);
    auto end = std::chrono::system_clock::now();
    float elapsed_time = std::chrono::duration<float, std::milli>(
        end - start).count();
    times.push_back(elapsed_time);
    std::cout << "elapsed time: " << elapsed_time << " ms" << std::endl;
    return ret;
}

static void print_stats(const std::string info, std::vector<float> &times)
{
    std::cout << info << " mean elapsed time: ";
    std::cout << std::fixed << std::setprecision(2) 
        << std::accumulate(times.begin(), times.end(), 0.0) 
            / times.size()
        << " ms" << std::endl;
}

static void print_speedup(const std::string info, std::vector<float> &times1, 
    std::vector<float> &times2)
{
    std::cout << info << " speedup: ";
    std::cout << std::fixed << std::setprecision(2) 
        << std::accumulate(times1.begin(), times1.end(), 0.0) 
            / std::accumulate(times2.begin(), times2.end(), 0.0) 
        << "x" << std::endl;
}

int main()
{
    std::vector<float> elapsed_bf_sequential_vec;
    std::vector<float> elapsed_bf_sequential_cv_vec;
    std::vector<float> elapsed_bf_parallel_vec;
    std::vector<float> elapsed_bf_parallel_cv_vec;

    double dist;
    bool equals;
    for (const auto & entry: fs::directory_iterator(input_dp.string()))
    {
#if defined(WIN32)
        if (entry.path().extension().string().compare(".db") == 0)
            continue;
#endif
        auto input_mat = cv::imread(entry.path().string(), 
            cv::IMREAD_GRAYSCALE);
        std::cout << "-- Image processed: " << entry.path().filename() << std::endl;

        auto out_bf_sequential_fp = output_bf_sequential_dp 
            / entry.path().filename();
        auto out_bf_sequential_cv_fp = output_bf_sequential_cv_dp 
            / entry.path().filename();
        auto out_bf_parallel_fp = output_bf_parallel_dp 
            / entry.path().filename();
        auto out_bf_parallel_cv_fp = output_bf_parallel_cv_dp 
            / entry.path().filename();

        // Bilateral Filter Sequential
        auto bf_sequential_mat = test("Bilateral Filter Sequential", 
            entry.path(), input_mat, bf_sequential, 
            elapsed_bf_sequential_vec);
        cv::imwrite(out_bf_sequential_fp.string(), bf_sequential_mat);

        // Bilateral Filter Sequential OpenCV
        auto bf_sequential_cv_mat = test("Bilateral Filter Sequential OpenCV", 
            entry.path(), input_mat, bf_sequential_cv, 
            elapsed_bf_sequential_cv_vec);
        cv::imwrite(out_bf_sequential_cv_fp.string(), bf_sequential_cv_mat);

        // Bilateral Filter Parallel
        auto bf_parallel_mat = test("Bilateral Filter Parallel", 
            entry.path(), input_mat, bf_parallel, elapsed_bf_parallel_vec);
        cv::imwrite(out_bf_parallel_fp.string(), bf_parallel_mat);

        // Bilateral Filter Parallel OpenCV
        auto bf_parallel_cv_mat = test("Bilateral Filter Parallel OpenCV", 
            entry.path(), input_mat, bf_parallel_cv, 
            elapsed_bf_parallel_cv_vec);
        cv::imwrite(out_bf_parallel_cv_fp.string(), bf_parallel_cv_mat);

        // Check if equals.
        equals = true;
        auto rows = bf_sequential_mat.rows;
        auto cols = bf_sequential_mat.cols;
        dist = cv::norm(bf_parallel_cv_mat, bf_sequential_cv_mat, cv::NORM_L2) 
            / static_cast<double>(rows * cols);
        if (dist > 0.001) equals = false;
        dist = cv::norm(bf_sequential_cv_mat, bf_sequential_mat, cv::NORM_L2)
            / static_cast<double>(rows * cols);
        if (dist > 0.001) equals = false;
#if CHECK_PARALLEL
        cv::Mat temp;
        cv::bitwise_xor(bf_parallel_mat, bf_sequential_mat, temp);
        if (cv::countNonZero(temp) > 0) equals = false;
#endif
        std::cout << (equals ? "CHECK: OK" : "CHECK: FAILED") << std::endl;
    }

    std::cout << "\n-- Average analysis: " << std::endl;
    print_stats("Bilateral Filter Sequential", elapsed_bf_sequential_vec);
    print_stats("Bilateral Filter Sequential OpenCV",
        elapsed_bf_sequential_cv_vec);
    print_stats("Bilateral Filter Parallel", elapsed_bf_parallel_vec);
    print_stats("Bilateral Filter Parallel OpenCV", elapsed_bf_parallel_cv_vec);

    std::cout << "\n-- Speedup analysis: " << std::endl;
    print_speedup("|Sequential        / Parallel OpenCV  |", 
        elapsed_bf_sequential_vec, elapsed_bf_parallel_cv_vec);
    print_speedup("|Sequential OpenCV / Parallel OpenCV  |", 
        elapsed_bf_sequential_cv_vec, elapsed_bf_parallel_cv_vec);
    print_speedup("|Sequential        / Sequential OpenCV|", 
        elapsed_bf_sequential_vec, elapsed_bf_sequential_cv_vec);
#if SPEEDUP_PARALLEL
    print_speedup("|Sequential        / Parallel         |", 
        elapsed_bf_sequential_vec, elapsed_bf_parallel_vec);
    print_speedup("|Sequential OpenCV / Parallel         |", 
        elapsed_bf_sequential_cv_vec, elapsed_bf_parallel_vec);
    print_speedup("|Parallel OpenCV   / Parallel         |", 
        elapsed_bf_parallel_cv_vec, elapsed_bf_parallel_vec);
#endif
}