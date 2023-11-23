#pragma once
#ifndef OVERLAY_HPP
#define OVERLAY_HPP

#include <string>
#include <opencv2/core.hpp/mat.hpp>
#include <image_processing/utility/constante.hpp>


/// @brief This function places an overlay corresponding to the alpha frame with a RGB color.
/// It can be used on images or inside the computation loop for video.
/// Doesn't handle saturation. Ptr manipulation.
/// @param background The frame used as foreground for overlaying 
/// @param alpha The frame used as layers for overlaying
/// @param outImage The matrix in which the result is saved
/// @param BGR The color to give to the overlay
/// @param cover The power of the overlaying
void colorBlending(
    const cv::Mat& background, const cv::Mat& alpha, cv::Mat& outImage, 
    const auto& BGR = clrs::RED, const double& cover = 0.25
);

/// @brief This function operates an alpha blendign using some Mat.
/// It can be used on images or inside the computation loop for video.
/// You must prepare the alpha frame out of the loop to avoid value pixel overload.
/// @param foreground The frame used as foreground for overlaying 
/// @param background The frame used as background
/// @param alpha The frame used as layers for overlaying
/// @param outImage The matrix in which the result is saved
void alphaBlending(
    const cv::Mat& foreground, 
    const cv::Mat& background, 
    const cv::Mat& alpha, 
    cv::Mat& outImage
);

/// @brief This function init the alpha frame.
/// @param alphaStr The string from which to load the alpah frame
/// @return Alpha frame ready to use
cv::Mat initAlphaFrame(const std::string& alphaStr);

/// @brief This function init the alpha frame.
/// @param alphaStr The string from which to load the alpah frame
/// @param frameResize To resize the alpha frame accordingly
/// @return Alpha frame ready to use
cv::Mat initAlphaFrame(
    const std::string& alphaStr, 
    const cv::Size& frameResize
);

#endif OVERLAY_HPP