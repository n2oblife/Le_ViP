

#pragma once 
#ifndef VIDEO_STREAM_HPP
#define VIDEO_STREAM_HPP

#include <thread> // to compile : -std=c++0x -pthrea (look cmake)
#include <opencv2/core.hpp>
#include <opencv2/videoio.hpp>

namespace vip
{
    /// @brief A class for I/O of video in another thread 
    /// to avoid blocking the main thread (hot path) 
    /// This class is inspired by the work of Adrian Rosebrock : https://pyimagesearch.com/2015/12/21/increasing-webcam-fps-with-python-and-opencv/
    /// @tparam M Represents the OpenCV's Mat class or any similar class
    template <class M = cv::Mat>
    class VideoStream
    {
        private:
            cv::VideoCapture _stream;
            std::shared_ptr<bool> _stop; // check init and role
            std::shared_ptr<M> _grabbedFrame;
        

        public:
            VideoStream(); // lot of overload
            VideoStream(const char &filename);
            VideoStream(const char &filename, const int apiPreference);
            VideoStream(const int index);
            VideoStream(const int index, const int apiPreference);
            ~VideoStream();
            cv::VideoCapture getVideoCap() const;
            void release();
    };

    template <class M = cv::Mat>
    class VideoStreamIO: public VideoStream
    {
        private:
            cv::VideoWriter _writer;
            M *_writtingFrame; // frame to write
        
        public:
            VideoStream(); // lot of overload

            ~VideoStreamIO();

    };

    template <class M>
    class BuffedVideoStream : public VideoStream
    {
        private :

        public :
    };

    template <class M>
    class BuffedVideoStreamIO : public VideoStreamIO
    {
        private :

        public :
            
    }
    
} // namespace vip

#endif // VIDEO_STREAM_HPP


template <class M>
vip::VideoStream<M>::VideoStream()
: _stream(0)
{
    _stream >> &(this->_grabbedFrame);
    _stopped = (_grabbedFrame.empty())
}

template <class M>
vip::VideoStream<M>::VideoStream(const char &filename)
: _stream(filename)
{
    _stream >> _grabbedFrame;
    _stopped = (_grabbedFrame.empty())
}

template <class M>
vip::VideoStream<M>::VideoStream(
    const char &filename, 
    const int apiPreference
) : _stream(filename, apiPreference)
{
    _stream >> _grabbedFrame;
    _stopped = (_grabbedFrame.empty())
}

template <class M>
vip::VideoStream<M>::VideoStream(const int index)
: _stream(index)
{
    _stream >> _grabbedFrame;
    _stopped = (_grabbedFrame.empty())
}

template <class M>
vip::VideoStream<M>::VideoStream(
    const int index, 
    const int apiPreference
) : _stream(index, apiPreference)
{
    _stream >> _grabbedFrame;
    _stopped = (_grabbedFrame.empty())
}

template <class M>
vip::VideoStream<M>::~VideoStream()
{
    _stream.~VideoCapture();
    delete _stopped;
}