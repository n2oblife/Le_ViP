#pragma once
#ifndef IMG_COORRECTION_HPP
#define IMG_COORRECTION_HPP

#include <image_processing/frame/roi.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>


namespace vip
{
    /// @brief This function changes the rendering using CLAHE algorithm  
    /// of an image by changing the color space from original using Lab 
    /// => more contrast
    /// @tparam M Represents the OpenCV's Mat class or any similar
    /// @param src_frame Frame to be changed
    /// @param out_frame Frame in BGR after some improvement
    /// @param isColor Tells if the source image is in BGR
    template <class M = cv::Mat>
    void lumenCorrection(
        M &src_frame,
        M &out_frame, 
        const bool& isColor = true
    );

    /// @brief This function changes the rendering using CLAHE algorithm  
    /// of an image by changing the color space from BGR using Lab 
    /// => more contrast
    /// @tparam M Represents the OpenCV's Mat class or any similar
    /// @param src_frame Frame to be changed
    /// @param out_frame Frame in BGR after some improvement
    template <class M>
    void lumenCorrectionBGR(
        M &src_frame,
        M &out_frame
    );

    /// @brief This function changes the rendering using CLAHE algorithm  
    /// of an image by changing the color space from Lab to BGR 
    /// => more contrast
    /// @tparam M Represents the OpenCV's Mat class or any similar
    /// @param src_frame Frame to be changed
    /// @param out_frame Frame in BGR after some improvement
    template <class M>
    void lumenCorrectionLab(
        M &src_frame,
        M &out_frame
    );

    /// @brief This function changes the rendering using CLAHE algorithm  
    /// of an vector of images by changing the color space from original 
    /// using Lab => more contrast
    /// @tparam M Represents the OpenCV's Mat class or any similar
    /// @tparam N Represents the size of the array
    /// @param src_frame Vector of frame to be changed
    /// @param out_frame Vector of frame in BGR after some improvement
    /// @param isColor Tells if the source images are in BGR
    template <class M, size_t N>
    inline void lumenCorrection(
        std::array<M, N> &src_vector,
        std::array<M, N> &out_vector,
        const bool& isColor = true
    )
    {
        for (int &i=0; i<src_vector.size(); i++)
            lumenCorrection(
                src_vector[i],
                out_vector[i],
                isColor
            );
    }

    /// @brief This function changes the rendering using CLAHE algorithm  
    /// of an vector of images by changing the color space from BGR using Lab 
    /// => more contrast
    /// @tparam M Represents the OpenCV's Mat class or any similar
    /// @tparam N Represents the size of the array
    /// @param src_frame Vector of frames to be changed
    /// @param out_frame Vetctor of frames in BGR after some improvement
    template <class M, size_t N>
    inline void lumenCorrectionBGR(
        std::array<M, N> &src_vector,
        std::array<M, N> &out_vector
    )
    {
        for (int &i=0; i<src_vector.size(); i++)
            lumenCorrectionBGR(
                src_vector[i],
                out_vector[i]
            );
    }

    /// @brief This function changes the rendering using CLAHE algorithm  
    /// of a vector of images by changing the color space from Lab to BGR 
    /// => more contrast
    /// @tparam M Represents the OpenCV's Mat class or any similar
    /// @tparam N Represents the size of the array
    /// @param src_frame Vector of frames to be changed
    /// @param out_frame Vector of frames in BGR after some improvement
    template <class M, size_t N>
    inline void lumenCorrectionLab(
        std::array<M, N> &src_vector,
        std::array<M, N> &out_vector
    )
    {
        for (int &i=0; i<src_vector.size(); i++)
            lumenCorrectionLab(
                src_vector[i],
                out_vector[i]
            );
    }

    /// @brief Denoising and eroding and dilating a frame
    /// @tparam M Represents the OpenCV's Mat class or any similar
    /// @param src_frame Frame to be denoised
    /// @param out_frame Frame denoised
    /// @param threshold Threshold after the gaussian blur
    /// @param niters Number of iterations for erosion and dilatation
    template <class M>
    void thresholdedGaussianBlur(
        M &src_frame, 
        M &out_frame,
        const double &threshold=45.,
        const int &niters=2
    );

    /// @brief Denoising and eroding and dilating a frame and turn it gray
    /// @tparam M Represents the OpenCV's Mat class or any similar
    /// @param src_frame Frame to be denoised
    /// @param out_frame Frame denoised
    /// @param threshold Threshold after the gaussian blur
    template <class M>
    void thresholdedGaussianBlurToGray(
        M &src_frame, 
        M &out_frame,
        const double& threshold = 45.
    );

    /// @brief Denoising and eroding and dilating a vector of frames
    /// @tparam M Represents the OpenCV's Mat class or any similar
    /// @tparam N Represents the size of the array
    /// @param src_vector Vector of frames to be denoised
    /// @param out_vector Vector of frames denoised
    /// @param threshold Threshold after the gaussian blur
    /// @param niters Number of iterations for erosion and dilatation
    /// @param turnGray Changes the final color space to gray
    template <class M, size_t N>
    inline void thresholdedGaussianBlur(
        std::array<M, N> &src_vector,
        std::array<M, N> &out_vector,
        const double& threshold=45.,
        const int& niters=2
    )
    {
        for (int &i=0; i<src_vector.size(); i++)
            thresholdedGaussianBlur(
                src_vector[i],
                out_vector[i],
                threshold,
                niters
            );
    }

    /// @brief Denoising and eroding and dilating a frame and turn it gray
    /// @tparam M Represents the OpenCV's Mat class or any similar
    /// @tparam N Represents the size of the array
    /// @param src_vector Vector of frames to be denoised
    /// @param out_vector Vector of frames denoised
    /// @param threshold Threshold after the gaussian blur
    template <class M, size_t N>
    inline void thresholdedGaussianBlurToGray(
        std::array<M, N> &src_vector, 
        std::array<M, N> &out_vector,
        const double& threshold = 45.
    )
    {
        for(int &i=0; i<src_vector.size(); i++)
            thresholdedGaussianBlurToGray(
                src_vector[i],
                out_vector[i],
                threshold
            );
    }
} // namespace vip

#endif // IMG_COORRECTION_HPP