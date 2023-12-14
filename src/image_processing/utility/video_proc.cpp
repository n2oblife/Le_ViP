#include <opencv2/core.hpp>
#include <opencv2/videoio.hpp>
#include <image_processing/frame/buffer.hpp>

// overload class for live processing and video processing ?
// for live processing no handling of the save ?
// use keyword explicit to resolve conversion problems
// use tuple to swape elements
// use 2^n int whenever possible to match binary archi
// add namespace everywhere

namespace vip
{
    template <class M = cv::Mat>
    class VideoProc
    {
        private :
            vip::Buffer<M> frameBuffer;
            std::shared_ptr<M> currentFrame; // might need more frame for parallel use of algos
            M originalFrame; // need to keep original frame for last overlay
            bool refresh;
            cv::VideoCapture

        public :
            VideoProc() // for futur will be constructed with external data
            void update();
            void process();
    };

    template <class M = cv::Mat>
    class VideoProcSaving: public VideoProc
    {
        private :
            char* writingFileName;

        public :
            VideoProc(char*) // for futur will be constructed with external data
    };

} // namespace vip
