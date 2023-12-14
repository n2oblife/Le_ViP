
#include <image_processing/frame/buffer.hpp>

// overload class for live processing and video processing ?
// for live processing no handling of the save ?
// use keyword explicit to resolve conversion problems
// use tuple to swape elements
// use 2^n int whenever possible to match binary archi

using namespace vip;

template <class M = cv::Mat>
Buffer::Buffer(
    const size_t buffSize
) :
_buffSize(buffSize),
_frameBuffer(std::array<M, buffSize>),
_framePtr(nullptr)
{}

template <class M = cv::Mat>
void Buffer::operator=(std::array<M, buffSize>)
{

}

template <class M = cv::Mat>
void vip::Buffer::


template <class M = cv::Mat>
M Buffer::operator[](const size_t& idx) const
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
    
