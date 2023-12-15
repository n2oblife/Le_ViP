
#include <image_processing/utility/buffer.hpp>

// overload class for live processing and video processing ?
// for live processing no handling of the save ?
// use keyword explicit to resolve conversion problems
// use tuple to swape elements
// use 2^n int whenever possible to match binary archi


template <class M>
vip::Buffer<M>::Buffer(
    const size_t bufferSize
) : _bufferSize(bufferSize),
    _frameBuffer(std::array<M, this->_bufferSize>()),
    _framePtr(nullptr)
{
    cv::AutoBuffer<M, bufferSize> buffer();
    buffer.
}

template <class M>
size_t vip::Buffer<M>::size() const
{
    return this->_bufferSize;
}

template <class M>
vip::Buffer<M> &vip::Buffer<M>::operator=(const int *otherBuffer)
{
    this->size();
    return *this;
}



template <class M>
M vip::Buffer<M>::operator[](const size_t& idx) const
{
    return vip::Buffer
}

Buffer::~Buffer()
{
    delete _buffSize;
}


// template <class M = cv::Mat>
// class Frame
// {
//     // consider some parameters statics ?

//     private :
//         std::shared_ptr<Buffer> frameBuffer;

//     public :
//         Frame();
//         void init();
//         void upda

//     private const size_t buffSize;
//     private std::array<M, buffSize> frameBuffer; 
//     private M currentFrame, processedFrame;
// };

