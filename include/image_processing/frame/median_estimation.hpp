#pragma once
#ifndef MEDIAN_ESTIMATION_HPP
#define MEDIAN_ESTIMATION_HPP

#include <opencv2/core.hpp>
#include <opencv2/videoio.hpp>

// TODO add some overlods inline


/// @brief Get median value of a list.
/// @tparam MEDIAN Type of the median element returned
/// @param Elements List of the elements
/// @return The median value of the list
template <typename MEDIAN>
MEDIAN getMedian(std::vector<MEDIAN> elements);

/// @brief Compute a frame with each pixel of median value.
/// @param vec A vector with Mat
/// @return A Mat with median pixel values
cv::Mat compute_median(std::vector<cv::Mat> vec);

/// @brief Compute a frame with each pixel of median value.
/// @tparam CAP_INPUT Can be any kind of InputArry from OpenCV
/// @param input Video file or video stream from which to compute the median frame
/// @param nFrames Number of frames to take into account from beggining 
/// @return A Mat with median pixel values 
template <typename CAP_INPUT>
cv::Mat computeMedianFrame(
    const CAP_INPUT& input,
    int& nFrames = 25
);

/// @brief Compute a frame with each pixel of median value.
/// @param cap VideoCapture element from which to compute the median frame
/// @param nFrames Number of frames to take into account from beggining
/// @return A Mat with median pixel values 
cv::Mat computeMedianFrame(
	cv::VideoCapture& cap,
    int& nFrames
);

/// @brief Compute a frame with each pixel of median value.
/// @tparam CAP_INPUT Can be any kind of InputArry from OpenCV
/// @param input Video file or video stream from which to compute the median frame
/// @param frame Avoid to init a nex frame
/// @param working Calculate the median along the time axis
/// @param nFrames Number of frames to take into account from beggining 
/// @return A Mat with median pixel values 
template <typename CAP_INPUT>
cv::Mat computeMedianFrame(
    const CAP_INPUT& input,
	cv::Mat& frame,
	const bool& working=true,
    int& nFrames = 25
);

/// @brief Compute a frame with each pixel of median value.
/// @param cap VideoCapture element from which to compute the median frame
/// @param frame Avoid to init a nex frame
/// @param nFrames Number of frames to take into account from beggining 
/// @param working Calculate the median along the time axis
/// @return A Mat with median pixel values 
cv::Mat computeMedianFrame(
	cv::VideoCapture& cap,
	cv::Mat& frame,
    int& nFrames,
	const bool& working=true
);

#endif // MEDIAN_ESTIMATION_HPP