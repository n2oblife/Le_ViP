#include <iostream>
#include <stdio.h>
#include <string>
#include <opencv2/opencv.hpp>
// #include <opencv2/core.hpp>
// #include <opencv2/videoio.hpp>
// #include <opencv2/highgui.hpp>

#include <cstdlib>

using namespace cv;
using namespace std;

int main(int argc, char* argv[])
{  
    //Open the default video camera
    VideoCapture cap(2);
    cap.set(CAP_PROP_BUFFERSIZE, 1); // internal buffer will now store only 1 frames

    // if not success, exit program
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

    Mat frame;
    Mat resized_frame;
    // cap >> frame;
    
    while (true)
    {
        cap >> frame;
        // If the frame is empty, break immediately
        if (frame.empty())
        break;

        // bool bSuccess = cap.read(frame); // read a new frame from video(can use operator to go faster)
        resize(frame, resized_frame, Size(640, 480), INTER_LINEAR);
        // The use of grab and retrieve will be useful when computing mulitple cameras
        // because retrieve syncrhonizes the decoding on multiple cameras

        //Breaking the while loop if the frames cannot be captured
        // if (bSuccess == false) 
        // {
        //     cout << "Video camera is disconnected" << endl;
        //     cin.get(); //Wait for any key press
        //     break;
        // }

        //show the frame in the created window
        imshow(window_name, resized_frame);

        //wait for for 10 ms until any key is pressed.  
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