#include <image_processing/segment_object.hpp>

#include "opencv2/imgproc.hpp"
#include "opencv2/videoio.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/video/background_segm.hpp"
#include <image_processing/utility.hpp>
#include <stdio.h>
#include <string>
#include <iostream>
using namespace std;
using namespace cv;

static void help(char** argv)
{
    printf("\n"
            "This program demonstrated a simple method of connected components clean up of background subtraction\n"
            "When the program starts, it begins learning the background.\n"
            "You can toggle background learning on and off by hitting the space bar.\n"
            "Call\n"
            "%s [video file, else it reads camera 0]\n\n", argv[0]);
}
static void refineSegments(const Mat& img, Mat& mask, Mat& dst)
{
    // Number of iterations for morphological operations
    int niters = 3;

    // Vectors to store contours and hierarchy information
    vector<vector<Point> > contours;
    vector<Vec4i> hierarchy;

    // Temporary Mat for intermediate processing
    Mat temp;

    // Apply morphological dilation to the input mask
    dilate(mask, temp, Mat(), Point(-1, -1), niters);

    // Apply morphological erosion to the dilated mask
    erode(temp, temp, Mat(), Point(-1, -1), niters * 2);

    // Apply morphological dilation to the eroded mask
    dilate(temp, temp, Mat(), Point(-1, -1), niters);

    // Find contours in the processed mask
    findContours(temp, contours, hierarchy, RETR_CCOMP, CHAIN_APPROX_SIMPLE);

    // Create an output Mat initialized with zeros
    dst = Mat::zeros(img.size(), CV_8UC3);

    // If there are no contours, return
    if (contours.size() == 0)
        return;

    // Iterate through all the top-level contours to find the largest one
    int idx = 0, largestComp = 0;
    double maxArea = 0;
    for (; idx >= 0; idx = hierarchy[idx][0])
    {
        const vector<Point>& c = contours[idx];

        // Calculate the area of the contour
        double area = fabs(contourArea(Mat(c)));

        // Update the information if the current area is larger than the maximum area
        if (area > maxArea)
        {
            maxArea = area;
            largestComp = idx;
        }
    }

    // Set the color for drawing contours (in BGR format)
    Scalar color(255, 255, 255);

    // Draw the largest contour on the output Mat
    drawContours(dst, contours, largestComp, color, FILLED, LINE_8, hierarchy);
}


int mainFunction(const string& input, Mat& out_frame, const int& threshold=10)
{
    // Initialize VideoCapture, set a flag for updating the background model, and parse command-line arguments
    VideoCapture cap;
    bool update_bg_model = true;

    // Determine the input source (camera or video file)
    determineInputCap(input, cap);

    // Check if the VideoCapture object is successfully opened
    if (!cap.isOpened())
    {
        cerr << "\nCan not open camera or video file\n";
        return -1;
    } 

    // Initialize Mat objects for temporary frame storage, background mask, and output frame
    Mat tmp_frame, bgmask;

    // Capture the first frame from the video source
    cap >> tmp_frame;
    if (tmp_frame.empty())
    {
        cerr << "\nCan not read data from the video source\n";
        return -1;
    }

    // Create windows for displaying the original video and segmented output
    namedWindow("Source", 1);
    namedWindow("Segmented", 1);
    namedWindow("Overlay", 1);

    // Create a BackgroundSubtractorMOG2 object and set a variance threshold
    Ptr<BackgroundSubtractorMOG2> bgsubtractor = createBackgroundSubtractorMOG2();
    bgsubtractor->setVarThreshold(threshold); // default value at definition of function

    // Main loop for processing video frames
    for (;;)
    {
        // Capture the current frame
        cap >> tmp_frame;

        // Break the loop if no more frames are available
        if (tmp_frame.empty())
        {
            cerr << "\nVideo flow broke, no more frame\n";
            return -1;
        }

        // Apply the background subtraction algorithm to obtain a background mask
        bgsubtractor->apply(tmp_frame, bgmask, update_bg_model ? -1 : 0);

        // Refine the segmentation using the custom refineSegments function
        refineSegments(tmp_frame, bgmask, out_frame);

        // Display the original video and segmented output
        imshow("video", tmp_frame);
        imshow("segmented", out_frame);

        // Wait for a key press (ESC to exit, spacebar to toggle background model updating)
        char keycode = (char)waitKey(30);
        if (keycode == 27)
            break;  // Break the loop if the ESC key is pressed
        if (keycode == ' ')
        {
            update_bg_model = !update_bg_model;
            printf("Learn background is in state = %d\n", update_bg_model);
        }
    }

    return 0;
}

int main(int argc, char** argv)
{
    // Initialize VideoCapture, set a flag for updating the background model, and parse command-line arguments
    VideoCapture cap;
    bool update_bg_model = true;
    CommandLineParser parser(argc, argv, "{help h||}{@input||}");

    // Display help if requested
    if (parser.has("help"))
    {
        help(argv);
        return 0;
    }

    // Determine the input source (camera or video file)
    string input = parser.get<std::string>("@input");
    if (input.empty())
        cap.open(0);  // If no input is provided, open the default camera (index 0)
    else
        cap.open(samples::findFileOrKeep(input));  // Open the specified video file

    // Check if the VideoCapture object is successfully opened
    if (!cap.isOpened())
    {
        printf("\nCan not open camera or video file\n");
        return -1;
    }

    // Initialize Mat objects for temporary frame storage, background mask, and output frame
    Mat tmp_frame, bgmask, out_frame;

    // Capture the first frame from the video source
    cap >> tmp_frame;
    if (tmp_frame.empty())
    {
        printf("can not read data from the video source\n");
        return -1;
    }

    // Create windows for displaying the original video and segmented output
    namedWindow("video", 1);
    namedWindow("segmented", 1);

    // Create a BackgroundSubtractorMOG2 object and set a variance threshold
    Ptr<BackgroundSubtractorMOG2> bgsubtractor = createBackgroundSubtractorMOG2();
    bgsubtractor->setVarThreshold(10);

    // Main loop for processing video frames
    for (;;)
    {
        // Capture the current frame
        cap >> tmp_frame;

        // Break the loop if no more frames are available
        if (tmp_frame.empty())
            break;

        // Apply the background subtraction algorithm to obtain a background mask
        bgsubtractor->apply(tmp_frame, bgmask, update_bg_model ? -1 : 0);

        // Refine the segmentation using the custom refineSegments function
        refineSegments(tmp_frame, bgmask, out_frame);

        // Display the original video and segmented output
        imshow("video", tmp_frame);
        imshow("segmented", out_frame);

        // Wait for a key press (ESC to exit, spacebar to toggle background model updating)
        char keycode = (char)waitKey(30);
        if (keycode == 27)
            break;  // Break the loop if the ESC key is pressed
        if (keycode == ' ')
        {
            update_bg_model = !update_bg_model;
            printf("Learn background is in state = %d\n", update_bg_model);
        }
    }

    return 0;
}
