#pragma once
#ifndef OVERLAY_HPP
#define OVERLAY_HPP

#include <opencv2/core.hpp>
#include <image_processing/utility/constante.hpp>

namespace vip
{
    /// @brief Return the number of parameters of a matrix
    /// @tparam M Represents the OpenCV's Mat class or any similar
    /// @param mat The matrix from which to have the number of pixels
    /// @return Number of pixels multiplied by channel, for alpha blending
    template <class M>
    inline int initNbreOfPixels(const M &mat)
    {
        return mat.rows * mat*cols * mat.channels
    }


    /// @brief This function places an overlay corresponding to the alpha frame with a RGB color.
    /// It can be used on images or inside the computation loop for video.
    /// Doesn't handle saturation. Ptr manipulation.
    /// @tparam M Represents the OpenCV's Mat class or any similar
    /// @param background The frame used as foreground for overlaying 
    /// @param alpha The frame used as layers for overlaying
    /// @param outImage The matrix in which the result is saved
    /// @param BGR The color to give to the overlay
    /// @param cover The power of the overlaying
    template <class M>
    void colorBlending(
        const M &background, 
        const M &alpha, 
        M &outImage, 
        const cv::Scalar &BGR = clrs::RED, 
        const double &cover = 0.25
    );

    /// @brief This function places an overlay corresponding to the alpha frame with a RGB color.
    /// It can be used on images or inside the computation loop for video.
    /// Doesn't handle saturation. Ptr manipulation.
    /// @tparam M Represents the OpenCV's Mat class or any similar
    /// @param background The frame used as foreground for overlaying 
    /// @param alpha The frame used as layers for overlaying
    /// @param outImage The matrix in which the result is saved
    /// @param BGR The color to give to the overlay
    /// @param cover The power of the overlaying
    template <class M>
    inline void colorBlendingTo32bits(
        const M &background, 
        const M &alpha, 
        M &outImage, 
        const cv::Scalar &BGR = clrs::RED, 
        const double &cover = 0.25
    )
    {
        // convert the images to 3 channels of 32 bit float
        foreground.convertTo(foreground, CV_32FC3);
        background.convertTo(background, CV_32FC3);

        colorBlending(
            background, 
            alpha, 
            outImage, 
            BGR, 
            cover
        );
    }



    /// @brief This function operates an alpha blendign using some Mat.
    /// It can be used on images or inside the computation loop for video.
    /// You must prepare the alpha frame out of the loop to avoid value pixel overload.
    /// @tparam M Represents the OpenCV's Mat class or any similar
    /// @param foreground The frame used as foreground for overlaying 
    /// @param background The frame used as background
    /// @param alpha The frame used as layers for overlaying
    /// @param outImage The matrix in which the result is saved
    template <class M>
    void alphaBlending(
        const M &foreground, 
        const M &background, 
        const M &alpha, 
        M &outImage
    );

    /// @brief This function init the alpha frame.
    /// @tparam M Represents the OpenCV's Mat class or any similar
    /// @param alphaStr The string from which to load the alpah frame
    /// @return Alpha frame ready to use
    template <class M>
    M &initAlphaFrame(const char &alphaStr);

    /// @brief This function init the alpha frame.
    /// @tparam M Represents the OpenCV's Mat class or any similar
    /// @param alphaStr The string from which to load the alpah frame
    /// @param frameResize To resize the alpha frame accordingly
    /// @return Alpha frame ready to use
    template <class M>
    M &initAlphaFrame(
        const char &alphaStr, 
        const cv::Size& frameResize
    );

} // namespace vip
#endif OVERLAY_HPP