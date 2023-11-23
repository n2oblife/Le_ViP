#include <string>
#include <iostream>
#include <sstream>
#include <opencv2/photo.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/video.hpp>
using namespace cv;
using namespace std;
const char* params
    = "{ help h         |           | Print usage }"
      "{ input          | vtest.avi | Path to a video or a sequence of image }"
      "{ algo           | MOG2      | Background subtraction method (KNN, MOG2) }";
int main(int argc, char* argv[])
{
    CommandLineParser parser(argc, argv, params);
    parser.about( "This program shows how to use background subtraction methods provided by "
                  " OpenCV. You can process both videos and images.\n" );
    if (parser.has("help"))
    {
        //print help information
        parser.printMessage();
    }
    //create Background Subtractor objects
    Ptr<BackgroundSubtractor> pBackSub;
    if (parser.get<String>("algo") == "MOG2")
        pBackSub = createBackgroundSubtractorMOG2();
    else
        pBackSub = createBackgroundSubtractorKNN();
    cout << "The input is " << parser.get<string>("input") << endl;
    string input = samples::findFileOrKeep(parser.get<string>("input") ) ;
    VideoCapture capture(input);
    if (!capture.isOpened()){
        //error in opening the video input
        cerr << "Unable to open: " << parser.get<String>("input") << endl;
        return 0;
    }
    Mat frame, gray_img, denoised_img,fgMask, resized_frame, resized_fgMask;
    while (true) {
        capture >> frame;
        if (frame.empty())
            break;
        // apply denoising
        // cvtColor(frame, gray_img, COLOR_BGR2GRAY); // maybe BGR
        // fastNlMeansDenoisingColored(frame, denoised_img, 3, 7, 21); // 3rd param is strenght of denoising
        //update the background model
        pBackSub->apply(frame, fgMask);
        //get the frame number and write it on the current frame
        rectangle(frame, cv::Point(10, 2), cv::Point(100,20),
                  cv::Scalar(255,255,255), -1);
        stringstream ss;
        ss << capture.get(CAP_PROP_POS_FRAMES);
        string frameNumberString = ss.str();
        putText(frame, frameNumberString.c_str(), cv::Point(15, 15),
                FONT_HERSHEY_SIMPLEX, 0.5 , cv::Scalar(0,0,0));

        resize(frame, resized_frame, Size(640, 480), INTER_LINEAR);
        resize(fgMask, resized_fgMask, Size(640, 480), INTER_LINEAR);

        //show the current frame and the fg masks
        imshow("Frame", resized_frame);
        imshow("FG Mask", resized_fgMask);
        //get the input from the keyboard
        int keyboard = waitKey(30);
        if (keyboard == 'q' || keyboard == 27)
            break;
    }
    return 0;
}