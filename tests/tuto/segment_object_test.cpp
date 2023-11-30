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

static void refineSegmentation(Mat& segMask, Mat& dst, const int& niters = 3)
{
    // Vectors to store contours and hierarchy information
    vector<vector<Point> > contours;
    vector<Vec4i> hierarchy;

    // Temporary Mat for intermediate processing
    Mat temp;

    // Apply morphological dilation to the input mask
    dilate(segMask, temp, Mat(), Point(-1, -1), niters);

    // Apply morphological erosion to the dilated mask
    erode(temp, temp, Mat(), Point(-1, -1), niters * 2);

    // Apply morphological dilation to the eroded mask
    dilate(temp, temp, Mat(), Point(-1, -1), niters);

    // Find contours in the processed mask
    findContours(temp, contours, hierarchy, RETR_CCOMP, CHAIN_APPROX_SIMPLE);

    // Create an output Mat initialized with zeros
    dst = Mat::zeros(segMask.size(), CV_8UC3);

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

