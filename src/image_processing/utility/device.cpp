// include the hpp file
// include hpp file for constante
#include <image_processing/utility/device.hpp>
#include <image_processing/utility/constante.hpp>

#include <string>
#include <opencv2/videoio.hpp>
#include <opencv2/core.hpp>


cv::VideoCapture initVideoCap(const std::string& input)
{
    cv::VideoCapture cap;
    // Determine the input source (camera or video file)
    if (input.empty())
        cap.open(0);  // If no input is provided, open the default camera (index 0)
    else if (input.size() == 1)
        cap.open(std::stoi(input));  // If the input is one string long, it is considered a number, opens the input index cam
    else
        cap.open(samples::findFileOrKeep(input));  // Open the specified video file
	if(!cap.isOpened())
		cerr << "Error opening video file\n";
        return break;
    return cap
}

cv::VideoCapture initVideoCap(const int& input)
{
    cv::VideoCapture cap;
    // Determine the input source (camera or video file)
    if (input.empty())
        cap.open(0);  // If no input is provided, open the default camera (index 0)
    else
        cap.open(input);
	if(!cap.isOpened())
		cerr << "Error opening video file\n";
        return break;
    return cap;
}

void initVideoCap(const std::string& input, cv::VideoCapture& cap)
{
    // Determine the input source (camera or video file)
    if (input.empty())
        cap.open(0);  // If no input is provided, open the default camera (index 0)
    else if (input.size() == 1)
        cap.open(std::stoi(input));  // If the input is one string long, it is considered a number, opens the input index cam
    else
        cap.open(samples::findFileOrKeep(input));  // Open the specified video file
	if(!cap.isOpened())
		cerr << "Error opening video file\n";
        return break;
}

void initVideoCap(const int& input, cv::VideoCapture& cap)
{
    // Determine the input source (camera or video file)
    if (input.empty())
        cap.open(0);  // If no input is provided, open the default camera (index 0)
    else
        cap.open(input);
	if(!cap.isOpened())
		cerr << "Error opening video file\n";
        return break;
}

void videoMaxFrame(
    const double& fps, 
    const size& frameSize, 
    const int& frameRate,
    int& maxFrame
)
{
    if (CV_VERSION_MAJOR >= 3)
    {
        // can use smrt/shrd ptr for optim
        maxFrame = constants::L_VIP_INFINITE;
    }
    // 1Gbytes = 1073741824bits
    maxFrame = (int) 2147483648 / (24 * frameRate * frameSize[0] * frameSize[1])
}

int videoMaxFrame(
    const double& fps, 
    const size& frameSize, 
    const int& frameRate
)
{
    if (CV_VERSION_MAJOR >= 3)
    {
        // can use smrt/shrd ptr for optim
        return cst::DBL_INFINITE;
    }
    // 1Gbytes = 1073741824bits
    return (int) 2147483648 / (24 * frameRate * frameSize[0] * frameSize[1])
}