#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/video.hpp>
using namespace cv;
using namespace std;

void opticalFlow(
    cv::Mat& previous_frame,
    cv::Mat& next_frame,
    cv::Mat& out_frame,
    const bool& isColor=false
)
{
    cv::Mat previous, next;
    if (isColor) 
    {
        cv::cvtColor(previous_frame, previous, cv::COLOR_BGR2GRAY);
        cv::cvtColor(next_frame, next, cv::COLOR_BGR2GRAY);
    }
    else  
    {
        previous_frame.copyTo(previous);
        next_frame.copyTo(next);
    }

    cv::Mat flow(previous.size(), CV_32FC2);
    cv::calcOpticalFlowFarneback(previous, next, flow, 0.5, 3, 15, 3, 5, 1.2, 0);
    // visualization
    cv::Mat flow_parts[2];
    cv::split(flow, flow_parts);
    cv::Mat magnitude, angle, magn_norm;
    cv::cartToPolar(flow_parts[0], flow_parts[1], magnitude, angle, true);
    cv::normalize(magnitude, magn_norm, 0.0f, 1.0f, NORM_MINMAX);
    angle *= ((1.f / 360.f) * (180.f / 255.f));
    //build hsv image
    cv::Mat _hsv[3], hsv, hsv8, bgr;
    _hsv[0] = angle;
    _hsv[1] = cv::Mat::ones(angle.size(), CV_32F);
    _hsv[2] = magn_norm;
    cv::merge(_hsv, 3, hsv);
    hsv.convertTo(hsv8, CV_8U, 255.0);
    cv::cvtColor(hsv8, bgr, COLOR_HSV2BGR);
    bgr.copyTo(out_frame);
}

// int main()
// {
//     VideoCapture capture(samples::findFile("../../../../datasets/puncture.mp4"));
//     if (!capture.isOpened()){
//         //error in opening the video input
//         cerr << "Unable to open file!" << endl;
//         return 0;
//     }
//     Mat frame1, prvs;
//     capture >> frame1;
//     cvtColor(frame1, prvs, COLOR_BGR2GRAY);
//     while(true){
//         Mat frame2, next;
//         capture >> frame2;
//         if (frame2.empty())
//             break;
//         cvtColor(frame2, next, COLOR_BGR2GRAY);
//         Mat flow(prvs.size(), CV_32FC2);
//         calcOpticalFlowFarneback(prvs, next, flow, 0.5, 3, 15, 3, 5, 1.2, 0);
//         // visualization
//         Mat flow_parts[2];
//         split(flow, flow_parts);
//         Mat magnitude, angle, magn_norm;
//         cartToPolar(flow_parts[0], flow_parts[1], magnitude, angle, true);
//         normalize(magnitude, magn_norm, 0.0f, 1.0f, NORM_MINMAX);
//         angle *= ((1.f / 360.f) * (180.f / 255.f));
//         //build hsv image
//         Mat _hsv[3], hsv, hsv8, bgr;
//         _hsv[0] = angle;
//         _hsv[1] = Mat::ones(angle.size(), CV_32F);
//         _hsv[2] = magn_norm;
//         merge(_hsv, 3, hsv);
//         hsv.convertTo(hsv8, CV_8U, 255.0);
//         cvtColor(hsv8, bgr, COLOR_HSV2BGR);
//         resize(bgr, bgr, Size(640, 480), INTER_LINEAR);
//         imshow("frame2", bgr);
//         int keyboard = waitKey(30);
//         if (keyboard == 'q' || keyboard == 27)
//             break;
//         prvs = next;
//     }
// }


int main()
{
    VideoCapture capture(samples::findFile("../../../../datasets/puncture.mp4"));
    if (!capture.isOpened()){
        //error in opening the video input
        cerr << "Unable to open file!" << endl;
        return 0;
    }
    Mat frame1, prvs;
    capture >> frame1;
    cvtColor(frame1, prvs, COLOR_BGR2GRAY);
    while(true){
        Mat frame2, next, bgr;
        capture >> frame2;
        if (frame2.empty())
            break;
        opticalFlow(frame1, frame2, bgr, true);
        resize(bgr, bgr, Size(640, 480), INTER_LINEAR);
        imshow("frame2", bgr);
        int keyboard = waitKey(30);
        if (keyboard == 'q' || keyboard == 27)
            break;
        frame2.copyTo(frame1);
    }
}