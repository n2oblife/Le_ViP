#pragma once
#ifndef DEVICE_HPP
#define DEVICE_HPP

#include <string>
#include <opencv2/video.hpp>
#include <opencv2/core.hpp>


/// @brief This function initiates a video capture
/// @param input The input to capture
/// @return VideoCapture object initiated
cv::VideoCapture initVideoCap(const std::string& input);

/// @brief This function initiates a video capture
/// @param input The input to capture
/// @return VideoCapture object initiated
cv::VideoCapture initVideoCap(const int& input);

/// @brief This function initiates a video capture
/// @param input The input to capture
/// @param cap The initiated video capture
void initVideoCap(const std::string& input, cv::VideoCapture& cap);

/// @brief This function initiates a video capture
/// @param input The input to capture
/// @param cap The initiated video capture
void initVideoCap(const std::string& input, cv::VideoCapture& cap);

/// @brief Gives the max of frames to be saved.
/// OpenCV 2.0's VideoWriter can handle max 2Gb of video to save.
/// OpenCV 3.0's VideoWriter seems to handle other file formats, such as mkv.
/// @param fps Video framerate
/// @param frameSize Size of a frame in pixel
/// @param maxFrame The max of frames to save
void videoMaxFrame(
    const double& fps, 
    const cv::Size& frameSize, 
    int& maxFrame
);

/// @brief Gives the max of frames to be saved.
/// OpenCV 2.0's VideoWriter can handle max 2Gb of video to save.
/// OpenCV 3.0's VideoWriter seems to handle other file formats, such as mkv.
/// @param fps Video framerate
/// @param frameSize Size of a frame in pixel
/// @return The max of frames to save
int videoMaxFrame(
    const double& fps, 
    const cv::Size& frameSize, 
    const int& frameRate
);

#endif // DEVICE_HPP