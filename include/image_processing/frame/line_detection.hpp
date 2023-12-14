#pragma once 
#ifndef LINE_DETECTION_HPP
#define LINE_DETECTION_HPP

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

/// @brief This function draws line based on the Hough algorithm
/// @param in_frame Frame to be computed
/// @param out_frame Frame on which to diplay the lines
/// @param cannyThreshold1 Threshold 1 from canny algo
/// @param cannyThreshold2 Threhsold 2 from canny algo
/// @param standard Use of standard Hough Transform or probabilistic one
void drawHoughLines(
    cv::Mat& in_frame,
    cv::Mat& out_frame,
    const double& cannyThreshold1 = 50,
    const double& cannyThreshold2 = 200,
    const bool& standard = false
);

/// @brief This function draws line based on the Hough algorithm
/// @param in_vector Vector of frames to be computed
/// @param out_vector Vector of frames on which to diplay the lines
/// @param cannyThreshold1 Threshold 1 from canny algo
/// @param cannyThreshold2 Threhsold 2 from canny algo
/// @param standard Use of standard Hough Transform or probabilistic one
inline void drawHoughLines(
    std::vector<cv::Mat>& in_vector,
    std::vector<cv::Mat>& out_vector,
    const double& cannyThreshold1 = 50,
    const double& cannyThreshold2 = 200,
    const bool& standard = false
) 
{
    for (int i=0; i<in_vector.size(); i++)
        drawHoughLines(
            in_vector[i], out_vector[i], 
            cannyThreshold1, cannyThreshold2, standard
        );
}

/// @brief This function draws line based on the Hough probabilistic algorithm
/// @param in_vector Vector of frames to be computed
/// @param out_vector Vector of frames on which to diplay the lines
/// @param cannyThreshold1 Threshold 1 from canny algo
/// @param cannyThreshold2 Threhsold 2 from canny algo
void drawHoughLinesProbabilistic(
    cv::Mat& in_frame,
    cv::Mat& out_frame,
    const double& cannyThreshold1 = 50,
    const double& cannyThreshold2 = 200
);

/// @brief This function draws line based on the Hough standard algorithm
/// @param in_vector Vector of frames to be computed
/// @param out_vector Vector of frames on which to diplay the lines
/// @param cannyThreshold1 Threshold 1 from canny algo
/// @param cannyThreshold2 Threhsold 2 from canny algo
void drawHoughLinesStandard(
    cv::Mat& in_frame,
    cv::Mat& out_frame,
    const double& cannyThreshold1 = 50,
    const double& cannyThreshold2 = 200
);

#endif // LINE_DETECTION_HPP