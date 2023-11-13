#include <iostream>
#include <opencv2/opencv.hpp>

// #include <opencv2/core.hpp>
// #include <opencv2/videoio.hpp>
// #include <opencv2/highgui.hpp>

using namespace std;
using namespace cv;

int main()
{
    // Open a camera
    // find a way to get the correct camera
    // might need a gui or cli 
    int deviceID = 2;             // 0 = open default camera
    int apiID = CAP_ANY;      // 0 = autodetect default API
    VideoCapture cap;
    cap.set(CAP_PROP_BUFFERSIZE, 1);
    cap.open(deviceID, apiID);

    // if not success, exit programm
    if (cap.isOpened() == false)
    {
        cout << "Cannot open the video camera" << endl;
        cin.get(); //wait for any key press
        return -1;
    }

    double dWidth = cap.get(CAP_PROP_FRAME_WIDTH); //get the width of frames of the video
    double dHeight = cap.get(CAP_PROP_FRAME_HEIGHT); //get the height of frames of the video

    cout << "Resolution of the video : " << dWidth << " x " << dHeight << endl;

    string window_name = "My Camera Feed";
    namedWindow(window_name); //create a window called "My Camera Feed"

    // Initialize the header which is constant
    // if a GPU is available it will connect to it
    // whatch out in case of accelerator use (except if built in gpu on cpu)
    // there is a shared memory flag that can be used if needed
    UMat frame;
    UMat resized_frame;

    while (true)
    {
        // Allocates the matrix' content with operator
        cap >> frame;
        resize(frame, resized_frame, Size(640, 480), INTER_NEAREST);

        // If the frame is empty, break immediately
        if (frame.empty())
        break;

        imshow(window_name, resized_frame);

        //wait for for 1 ms until any key is pressed.  
        //If the 'Esc' key is pressed, break the while loop.
        //If the any other key is pressed, continue the loop 
        //If any key is not pressed withing 10 ms, continue the loop 
        if (waitKey(1) == 27)
        {
            cout << "Esc key is pressed by user. Stoppig the video" << endl;
            break;
        }
    }

    return 0;  
}