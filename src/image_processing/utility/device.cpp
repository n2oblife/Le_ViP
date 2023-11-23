// include the hpp file
// include hpp file for constante
#include <image_processing/utility/device.hpp>

#include <string>
#include <opencv2/videoio.hpp>
#include <opencv2/core/version.hpp>

using namespace std;
using namespace cv;

void determineInputVideoCap(const string& input, VideoCapture& cap)
{
    // Determine the input source (camera or video file)
    if (input.empty())
        cap.open(0);  // If no input is provided, open the default camera (index 0)
    else if (input.size() == 1)
        cap.open(stoi(input));  // If the input is one string long, it is considered a number, opens the input index cam
    else
        cap.open(samples::findFileOrKeep(input));  // Open the specified video file
}

/*max of 2Gb of video to save 
OpenCV 3.0's VideoWriter seems to handle other file formats, such as mkv*/
void videoMaxFrame(
    const double& fps, 
    const size& frameSize, 
    const int& frameRate,
    int& maxFrame)
{
    if (CV_VERSION_MAJOR >= 3)
    {
        // can use smrt/shrd ptr for optim
        maxFrame = L_VIP_INFINITE;
    }
    // 1Gbytes = 1073741824bits
    maxFrame = 2147483648 / (24 * frameRate * frameSize[0] * frameSize[1])
}

int videoMaxFrame(
    const double& fps, 
    const size& frameSize, 
    const int& frameRate)
{
    if (CV_VERSION_MAJOR >= 3)
    {
        // can use smrt/shrd ptr for optim
        return L_VIP_INFINITE;
    }
    // 1Gbytes = 1073741824bits
    return 2147483648 / (24 * frameRate * frameSize[0] * frameSize[1])
}