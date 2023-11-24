#pragma once
#ifndef MEDIAN_ESTIMATION_HPP
#define MEDIAN_ESTIMATION_HPP

#include <opencv2/core.hpp>
#include <iostream>

// can be an int or a str
template <typename CAP_INPUT>;
template <typename MEDIAN>;

/// @brief Get median value of a list.
/// @param elements List of the elements
/// @return The median value of the list
MEDIAN getMedian(std::vector<MEDIAN> elements);

/// @brief Compute a frame with each pixel of median value.
/// @param vec A vector with Mat
/// @return A Mat with median pixel values
cv::Mat compute_median(std::vector<cv::Mat> vec);

/// @brief Compute a frame with each pixel of median value.
/// @param input Video file or video stream from which to compute the median frame
/// @param nFrames Number of frames to take into account from beggining 
/// @return A Mat with median pixel values 
cv::Mat computeMedianFrame(
    const CAP_INPUT& input,
    int& nFrames = 25
);

#endif // MEDIAN_ESTIMATION_HPP