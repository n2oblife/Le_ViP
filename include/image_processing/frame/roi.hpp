#pragma once
#ifndef ROI_HPP
#define ROI_HPP

#include <opencv2/core.hpp>

namespace vip
{
    /// @brief This class 
    /// @tparam M 
    /// @tparam N 
    template <class M = cv::Mat, size_t N>
    class ROI
    {
        private:
            std::array<M, N> _roiArray; // should be an array of ptr 
            std::array<std::tuple<cv::Point, cv::Size>, N> _infos

        public :
            ROI(); // minimum is 1
            ROI(size_t arraySize);
            ~ROI();
            ROI &at(size_t &index);
            ROI &operator[](size_t &index);
            void splitFrameToROI(M &frame);
            size_t &size() const;


    };

} // namespace vip

#endif // ROI_HPP