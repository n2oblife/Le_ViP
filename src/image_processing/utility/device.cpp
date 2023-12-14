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
		std::cerr << "Error opening video file\n";
        return;
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
		std::cerr << "Error opening video file\n";
        return;
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
		std::cerr << "Error opening video file\n";
        return ;
}

void initVideoCap(const int& input, cv::VideoCapture& cap)
{
    // Determine the input source (camera or video file)
    if (input.empty())
        cap.open(0);  // If no input is provided, open the default camera (index 0)
    else
        cap.open(input);
	if(!cap.isOpened())
		std::cerr << "Error opening video file\n";
        return ;
}


/// @brief A function to init the VideoWriter and save frames
/// @param output Place to save the frames
/// @param api API to use
/// @param codec fourcc codec, a 4 char string in CAPITAL
/// @param fps Frame rate
/// @param size Size of the frame
/// @param isColor Choose to save in color or not
/// @return 
cv::VideoWriter initVideoWriter(
    const std::string& output,
    const int& api = 0, // cv::CAP_ANY
    const std::string& codec = "MJPG", 
    const double& fps=30.,
    const cv::Size size = cv::Size(1920, 1080),
    const bool& isColor = true 
)
{
    // Init the video writer
    cv::VideoWriter writer = cv::VideoWriter(
        output, api,
        cv::VideoWriter::fourcc(
            codec[0], codec[1], codec[2], codec[3]
        ),
        fps, size, isColor
    );
    // check if we succeeded
    if (!writer.isOpened()) {
        std::cerr << "Could not open the output video file for write\n";
        return ;
    }
    return writer;
}

void videoMaxFrame(
    const double& fps, 
    const cv::Size& frameSize, 
    const int& frameRate,
    int& maxFrame
)
{
    if (CV_VERSION_MAJOR >= 3)
    {
        // can use smrt/shrd ptr for optim
        maxFrame = cst::L_VIP_INFINITE;
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
    return std::div_t( 2147483648, (24 * frameRate * frameSize[0] * frameSize[1])).quot
}

int videoMaxFrame(
    const cv::VideoCapture& cap
)
{
    if (cv::CV_VERSION_MAJOR >= 3)
    {
        // can use smrt or shrd ptr for optim
        return cst::DBL_INFINITE;
    }
    // Get the video info
    const int fps = cap.get(cv::CAP_PROP_FPS);
    const int bitrate = cap.get(cv::CAP_PROP_BITRATE);
    // 1Gbytes = 1073741824bits
    // TODO check if cast is ok
    return (int) std::div_t(2147483648, bitrate).quot * fps;
}

int videoMaxFrame(
    const cv::VideoWriter& writer
)
{
    if (cv::CV_VERSION_MAJOR >= 3)
    {
        // can use smrt or shrd ptr for optim
        return cst::DBL_INFINITE;
    }
    // Get the video info
    const int fps = writer.get(cv::CAP_PROP_FPS);
    const int bitrate = writer.get(cv::CAP_PROP_BITRATE);
    // 1Gbytes = 1073741824bits
    // TODO check if cast is ok
    return (int) std::div_t(2147483648, bitrate).quot * fps;
}