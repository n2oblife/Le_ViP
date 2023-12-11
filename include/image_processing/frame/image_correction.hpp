#pragma once
#ifndef IMG_COORRECTION_HPP
#define IMG_COORRECTION_HPP

#include <opencv2/core.hpp>

/// @brief This function changes the rendering using CLAHE algorithm  
/// of an image by changing the color space from original using Lab 
/// => more contrast
/// @param src_frame Frame to be changed
/// @param out_frame Frame in BGR after some improvement
/// @param isColor Tells if the source image is in BGR 
void lumenCorrection(
    cv::Mat& src_frame,
    cv::Mat& out_frame, 
    const bool& isColor = true
);

/// @brief This function changes the rendering using CLAHE algorithm  
/// of an image by changing the color space from BGR using Lab 
/// => more contrast
/// @param src_frame Frame to be changed
/// @param out_frame Frame in BGR after some improvement
void lumenCorrectionBGR(
    cv::Mat& src_frame,
    cv::Mat& out_frame
);

/// @brief This function changes the rendering using CLAHE algorithm  
/// of an image by changing the color space from Lab to BGR 
/// => more contrast
/// @param src_frame Frame to be changed
/// @param out_frame Frame in BGR after some improvement
void lumenCorrectionLab(
    cv::Mat& src_frame,
    cv::Mat& out_frame
);

/// @brief This function changes the rendering using CLAHE algorithm  
/// of an vector of images by changing the color space from original 
/// using Lab => more contrast
/// @param src_frame Vector of frame to be changed
/// @param out_frame Vector of frame in BGR after some improvement
/// @param isColor Tells if the source images are in BGR
inline void lumenCorrection(
    std::vector<cv::Mat> src_vector,
    std::vector<cv::Mat> out_vector,
    const bool& isColor = true
)
{
    for (int i=0; i<src_vector.size(); i++)
        lumenCorrection(
            src_vector[i],
            out_vector[i],
            isColor
        );
}

/// @brief This function changes the rendering using CLAHE algorithm  
/// of an vector of images by changing the color space from BGR using Lab 
/// => more contrast
/// @param src_frame Vector of frames to be changed
/// @param out_frame Vetctor of frames in BGR after some improvement
inline void lumenCorrectionBGR(
    std::vector<cv::Mat> src_vector,
    std::vector<cv::Mat> out_vector
)
{
    for (int i=0; i<src_vector.size(); i++)
        lumenCorrectionBGR(
            src_vector[i],
            out_vector[i]
        );
}

/// @brief This function changes the rendering using CLAHE algorithm  
/// of a vector of images by changing the color space from Lab to BGR 
/// => more contrast
/// @param src_frame Vector of frames to be changed
/// @param out_frame Vector of frames in BGR after some improvement
inline void lumenCorrectionLab(
    std::vector<cv::Mat> src_vector,
    std::vector<cv::Mat> out_vector
)
{
    for (int i=0; i<src_vector.size(); i++)
        lumenCorrectionLab(
            src_vector[i],
            out_vector[i]
        );
}

/// @brief Denoising and eroding and dilating a frame
/// @param src_frame Frame to be denoised
/// @param out_frame Frame denoised
/// @param threshold Threshold after the gaussian blur
/// @param niters Number of iterations for erosion and dilatation
void thresholdedGaussianBlur(
    cv::Mat& src_frame, 
    cv::Mat& out_frame,
    const double& threshold=45.,
    const int& niters=2
);

/// @brief Denoising and eroding and dilating a frame and turn it gray
/// @param src_frame Frame to be denoised
/// @param out_frame Frame denoised
/// @param threshold Threshold after the gaussian blur
void thresholdedGaussianBlurToGray(
    cv::Mat& src_frame, 
    cv::Mat& out_frame,
    const double& threshold = 45.
);

/// @brief Denoising and eroding and dilating a vector of frames
/// @param src_vector Vector of frames to be denoised
/// @param out_vector Vector of frames denoised
/// @param threshold Threshold after the gaussian blur
/// @param niters Number of iterations for erosion and dilatation
/// @param turnGray Changes the final color space to gray
inline void thresholdedGaussianBlur(
    std::vector<cv::Mat>& src_vector,
    std::vector<cv::Mat>& out_vector,
    const double& threshold=45.,
    const int& niters=2
)
{
    for (int i=0; i<src_vector.size(); i++)
        thresholdedGaussianBlur(
            src_vector[i],
            out_vector[i],
            threshold,
            niters
        );
}

/// @brief Denoising and eroding and dilating a frame and turn it gray
/// @param src_vector Vector of frames to be denoised
/// @param out_vector Vector of frames denoised
/// @param threshold Threshold after the gaussian blur
inline void thresholdedGaussianBlurToGray(
    std::vector<cv::Mat>& src_vector, 
    std::vector<cv::Mat>& out_vector,
    const double& threshold = 45.
)
{
    for(int i=0; i<src_vector.size(); i++)
        thresholdedGaussianBlurToGray(
            src_vector[i],
            out_vector[i],
            threshold
        );
}

#endif // IMG_COORRECTION_HPP