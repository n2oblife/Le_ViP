#include <image_processing/utility.hpp>

#include <string>
#include <opencv2/videoio.hpp>

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

void initializeWriter()
{
    
}

void saveVideo(
    VideoCapture& inputCap, 
    VideoWriter& writer,
    const string& filename = "./video.avi",
    const double& fps = 25.0, 
    const string& str_codec = '',
    const bool& show = false)
{
    if (! str_codec.size(4))
    {
        const int& codec = 0;
    }
    const int& codec = VideoWriter::fourcc(
        str_codec[0],
        str_codec[1],
        str_codec[2],
        str_codec[3]);

    Mat src;
    // get one frame from camera to know frame size and type
    inputCap >> src;
    // check if we succeeded
    if (src.empty()) {
        cerr << "ERROR! blank frame grabbed\n";
        return;
    }
    bool isColor = (src.type() == CV_8UC3); // check if needed as param
    writer.open(filename, codec, fps, src.size(), isColor);
    // check if we succeeded
    if (!writer.isOpened()) {
        cerr << "Could not open the output video file for write\n";
        return;
    }
    //--- GRAB AND WRITE LOOP
    cout << "Writing videofile: " << filename << endl
         << "Press any key to terminate" << endl;
    for (;;)
    {
        // check if we succeeded
        if (!cap.read(src)) {
            cerr << "ERROR! blank frame grabbed\n";
            break;
        }
        // encode the frame into the videofile stream
        writer.write(src);
        if (show)
        {
            // show live and wait for a key with timeout long enough to show images
            imshow("Live", src);
        }
        if (waitKey(5) == 27)
            break;
    }
    // the videofile will be closed and released automatically in VideoWriter destructor
    return 0;
}

void videoMaxFrame()
{}