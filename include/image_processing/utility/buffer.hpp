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
    class Buffer // think about a generic array class with heredity for buffers
    {

        private :
            static const size_t _bufferSize; 
            std::array<M, _bufferSize> _frameBuffer;
            M *_framePtr;

        public:
            Buffer(const size_t bufferSize);
            size_t size() const; // needed ? faster with direct access ?
            Buffer &operator=(const int *otherBuffer);
            M at(size_t &idx) const;
            M operator[](size_t &idx) const; // return the value itself for less memory usage
            bool operator==(const Buffer &buff1, const Buffer &buff2) const;
            ~Buffer();
    };


} // namespace vip


#endif // BUFFER_HPP