#pragma once
#ifndef REFINING_HPP
#define REFINING_HPP


#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

namespace vip
{
    
    /// @brief Morpholocial transformations adapted for segmentation refinement
    /// @tparam M Represents the OpenCV's Mat class or any similar
    /// @param src_frame Segmentation mask to be morpholised
    /// @param niters Number of iterations for the operations
    template <class M>
    inline void refineMorphologics(
        M &src_frame,
        const int &niters = 3
    )
    {
        // Apply morphological dilation to the input mask
        cv::dilate(src_frame, src_frame, cv::Mat(), cv::Point(-1, -1), niters*2);
        // Apply morphological erosion to the dilated mask
        cv::erode(src_frame, src_frame, cv::Mat(), cv::Point(-1, -1), niters);
        // Apply morphological dilation to the eroded mask
        cv::dilate(src_frame, src_frame, cv::Mat(), cv::Point(-1, -1), niters);
        cv::erode(src_frame, src_frame, cv::Mat(), cv::Point(-1, -1), 1); 
    }

    /// @brief Morpholocial transformations adapted for segmentation refinement
    /// @tparam M Represents the OpenCV's Mat class or any similar
    /// @param src_frame Segmentation mask to be morpholised
    /// @param out_frame Output of the source mask refined
    /// @param niters Number of iterations for the operations
    template <class M>
    inline void refineMorphologics(
        M &src_frame,
        M &out_frame,
        const int &niters = 3
    )
    {
        // Apply morphological dilation to the input mask
        cv::dilate(src_frame, out_frame, cv::Mat(), cv::Point(-1, -1), niters*2);
        // Apply morphological erosion to the dilated mask
        cv::erode(out_frame, out_frame, cv::Mat(), cv::Point(-1, -1), niters);
        // Apply morphological dilation to the eroded mask
        cv::dilate(out_frame, out_frame, cv::Mat(), cv::Point(-1, -1), niters);
        cv::erode(out_frame, out_frame, cv::Mat(), cv::Point(-1, -1), 1); 
    }

    /// @brief Morpholocial transformations adapted for segmentation refinement
    /// @tparam M Represents the OpenCV's Mat class or any similar
    /// @tparam NC Number max of contours to be stored 
    /// @param src_vector Vector of segmentation masks to be morpholised
    /// @param niters Number of iterations for the operations
    template <class M, size_t N>
    inline void refineMorphologics(
        std::array<M,N> &src_vector,
        const int &niters = 3
    )
    {
        for (int i=0; i<N; i++)
            refineMorphologics(src_vector[i], niters);
    }

    /// @brief Morpholocial transformations adapted for segmentation refinement
    /// @tparam M Represents the OpenCV's Mat class or any similar
    /// @tparam NC Number max of contours to be stored 
    /// @param src_vector Vector of segmentation masks to be morpholised
    /// @param out_vector Vector of the output of the refined mask
    /// @param niters Number of iterations for the operations
    template <class M, size_t N>
    inline void refineMorphologics(
        std::array<M,N> &src_vector,
        std::array<M, N> &out_vector,
        const int &niters = 3
    )
    {
        for (int i=0; i<N; i++)
            refineMorphologics(src_vector[i], out_vector[i] niters);
    }

    /// @brief Computes the largest contour from a black and white frame in
    /// a gray colorspace
    /// @tparam M Represents the OpenCV's Mat class or any similar
    /// @tparam NC Number max of contours to be stored
    /// @param seg_mask Segmentation mask to be contoured
    /// @param largestContour Vector of the contour's points
    /// @param niters Number of iterations used for morphological transformation
    template <class M, size_t N>
    void getLargestContour(
        M &seg_mask,
        std::array<M,N> &largest_contour,
        const int &niters = 3
    );


    /// @brief Computes the largest contour from a vectors of black and white frames in
    /// a gray colorspace    
    /// @tparam M Represents the OpenCV's Mat class or any similar
    /// @tparam NC Number max of contours to be stored
    /// @tparam NP Number maximal of points to be stored
    /// @param src_vector Vector of segmentation masks to be contoured
    /// @param largestContour_vector Vector of vectors of the contours' points
    /// @param niters Number of iterations used for morphological transformation
    template < class M, size_t NC, size_t NP>
    inline void getLargestContour(
        std::array<M,NC> &src_vector,
        std::array<std::array<cv::Point, NP>, NC> &largest_contour_vector,
        const int& niters = 3
    )
    {
        for (int i=0; i<src_vector.size(); i++)
            getLargestContour(
                src_vector[i],
                largest_contour_vector[i],
                niters
            );
    }

    /// @brief Draw the largest contour on the image.
    /// @param seg_mask Segmentation image with multiple elements
    /// @param out_frame Frame on which the largest element only will be draw
    /// @param largestContour Return the largest element point
    /// @param color Color to draw the largest element
    /// @param niters Number of iterations used for morphological transformation
    template <class M, size_t NC>
    inline void drawLargestContour(
        M &seg_mask,
        M &out_frame,
        std::array<cv::Point, NC> &largest_contour,
        const cv::Scalar &color = cv::Scalar(255,255,255),
        const int &niters = 3
    )
    {
        getLargestContour(
            seg_mask, largest_contour, niters
        );
        cv::drawContours(out_frame, std::vector(largest_contour),0, color, cv::FILLED, cv::LINE_8);
    }

    /// @brief Draw the largest contour on the vecotr of images.
    /// @tparam M Represents the OpenCV's Mat class or any similar
    /// @tparam NC Number max of contours to be stored
    /// @tparam NP Number maximal of points to be stored
    /// @param src_vector Vector of segmentation images with multiple elements
    /// @param out_vector Vector of frames on which the largest element only will be draw
    /// @param largestContour_vector Return a vector of the largest element point
    /// @param color Color to draw the largest element
    /// @param niters Number of iterations used for morphological transformation
    template <class M, size_t NC, size_t NP>
    inline void drawLargestContour(
        std::array<M, NC> &src_vector,
        std::array<M, NC> &out_vector,
        std::array<std::array<cv::Point, NP>, NC> &largestContour_vector,
        const cv::Scalar &color = cv::Scalar(255,255,255),
        const int &niters = 3
    )
    {
        for (int i=0; i<src_vector.size(); i++)
            drawLargestContour(
                src_vector[i], out_vector[i], largestContour_vector[i], color, niters
            );
    }

    /// @brief Draw the largest contour on the vecotr of images.
    /// @tparam M Represents the OpenCV's Mat class or any similar
    /// @tparam NC Number max of contours to be stored
    /// @tparam NP Number maximal of points to be stored
    /// @param src_vector Vector of segmentation images with multiple elements
    /// @param out_vector Vector of frames on which the largest element only will be draw
    /// @param largestContour_vector Return a vector of the largest element point
    /// @param color_vetor A vector of colors to draw the largest element
    /// @param niters Number of iterations used for morphological transformation
    template <class M, size_t NC, size_t NP>
    inline void drawLargestContour(
        std::array<M, NC>& src_vector,
        std::array<M, NC>& out_vector,
        std::array<std::array<cv::Point, NP>, NC>& largest_contour_vector,
        const std::array<cv::Scalar, NC>& color_vector,
        const int& niters = 3
    )
    {
        for (int i=0; i<src_vector.size(); i++)
            drawLargestContour(
                src_vector[i], out_vector[i], largest_contour_vector[i], color_vector[i], niters
            );
    }

} // namespace vip


#endif // REFINING_HPP