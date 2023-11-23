// This file defines useful constant used in this project
#pragma once
#if !defined(L_VIP_CONSTANT_HPP)
#define L_VIP_CONSTANT_HPP 1

#include <opencv2/core.hpp>
#include <limits>


/// @brief Versions of the lib
namespace vrs
{
    const float L_VIP_VERSION=0.1;
} // namespace version

/// @brief Constantes to be used in the LeVIP lib
namespace cst
{
    // 32bit int infinite
    const int INT_INFINITE = std::numeric_limits<int>::max();
    // 32bit double infinite
    const double DBL_INFINITE = std::numeric_limits<double>::infinity();
} // namespace constants
 
/// @brief Colors to be used in the lib
namespace clrs
{
    const auto BLACK  = cv::Scalar(0, 0, 0);
    const auto WHITE  = cv::Scalar(255, 255, 255);
    const auto RED    = cv::Scalar(0, 0, 255);
    const auto LIME   = cv::Scalar(0, 255, 0);
    const auto BLUE   = cv::Scalar(255, 0, 0);
    const auto YELLOW = cv::Scalar(0, 255, 255);
    const auto CYAN   = cv::Scalar(255, 255, 0);
    const auto MAGENTA= cv::Scalar(255, 0, 255);
    const auto GRAY   = cv::Scalar(128, 128, 128);
    const auto MAROON = cv::Scalar(0, 0, 128);
    const auto OLIVE  = cv::Scalar(0, 128, 128);
    const auto GREEN  = cv::Scalar(0, 128, 0);
    const auto PURPLE = cv::Scalar(128, 0, 128);
    const auto TEAL   = cv::Scalar(128, 128, 0);
    const auto NAVY   = cv::Scalar(128, 0, 0);
} // namespace colors


#endif // L_VIP_CONSTANT_HPP