#pragma once 
#ifndef BUFFER_HPP
#define BUFFER_HPP

#include <opencv2/core.hpp>

namespace vip
{
    /// @brief A buffer class to store and manage data storage of the frames
    /// along algorithm and enable work on different thread in futur than 
    /// @tparam M Represents the OpenCV's Mat class or any similar
    template <class M = cv::Mat>
    class Buffer
    {
        private :
            const size_t _buffSize; 
            std::array<M, buffSize> _frameBuffer;
            M* _framePtr;

        public:
            Buffer(size_t);
            void operator=(std::array<M, buffSize>);
            M at(size_t &idx) const;
            M operator[](size_t &idx) const;
            bool operator==(const Buffer &buff1, const Buffer &buff2) const;
            ~Buffer();
    };


} // namespace vip


#endif // BUFFER_HPP