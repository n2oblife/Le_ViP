#include <image_processing/frame/refining.hpp>


// TODO write a version which handles both size clipping with two percents 


template <class M, size_t N>
void getLargestContour(
    M &seg_mask,
    std::array<cv::Point, N> &largest_contour,
    const int& niters = 3
)
{
    // Temporary Mat for intermediate processing, we assume it's already gray
    cv::Mat temp(seg_mask);

    // Vectors to store contours and hierarchy information
    std::vector< std::vector<cv::Point> > contours;
    std::vector<cv::Vec4i> hierarchy;

    refineMorphologics(temp, niters);

    // Find contours in the processed mask
    cv::findContours(temp, contours, hierarchy, cv::RETR_CCOMP, cv::CHAIN_APPROX_SIMPLE);

    // If there are no contours, return
    if (contours.size() == 0)
        return ;

    // Iterate through all the top-level contours to find the largest one
    // (can be used for scnd one too)
    int idx = 0, largestComp = 0; // scndLargest = 0;
    double maxArea = 0;
    for (; idx >= 0; idx = hierarchy[idx][0])
    {
        const std::vector<cv::Point>& c = contours[idx];

        // Calculate the area of the contour
        double area = fabs(cv::contourArea(cv::Mat(c)));

        // Update the information if the current area is larger than the maximum area
        if (area > maxArea)
        {
            maxArea = area;
            // scndLargest = largestComp; 
            largestComp = idx;
        }
    }    
    largest_contour = contours[largestComp];
}


template <class M, size_t NC, size_t NP>
void drawLargestContourCheating(
    M &seg_mask,
    M &out_frame,
    std::array<cv::Point, NC> &largest_contour,
    double &last_big,
    const double &min_threshold,
    const double &max_threshold,
    const cv::Scalar &color = cv::Scalar(255,255,255),
    const int &niters = 3   
)
{
    getLargestContour(seg_mask, largest_contour, niters);

    // cv::drawContours(dst, contours, largestComp, color, cv::FILLED, cv::LINE_8, hierarchy);
    // use a vector for last_big
    if (maxArea > (max_threshold * last_big) )
    {
        // Draw the largest contour on the output Mat
        // cv::fillPoly(dst, contours.at(largestComp), color, cv::LINE_8);
        cv::drawContours(dst, std::array(), largestComp, color, cv::FILLED, cv::LINE_8, hierarchy);
        if (24000. > maxArea) last_big = maxArea;
        refresh = true;
    }    
    elif (maxArea < (min_threshold * last_big))
    {
        // Draw largest contour 
    }
    else 
    {
        last_mat.copyTo(dst);
        refresh = false;
    }

    cv::drawContours(out_frame, std::vector(largest_contour),0, color, cv::FILLED, cv::LINE_8);
}

template <class M, size_t NC, size_t NP>
void drawLargestContourCheating(
    M &printing_frame,
    std::array<cv::Point, NC> largest_contour,
    const double& min_threshold,
    const double& max_threshold,
    const cv::Scalar& color = cv::Scalar(255,255,255)
)
{

}

// TODO extract the main points after finishing the PCA
std::tuple<cv::Point, cv::Point, double> refineSegmentationProto(
    cv::Mat& segMask, 
    cv::Mat& dst, 
    double& last_big,
    const cv::Mat& last_mat,
    bool& refresh,
    const bool& print_vectors = false,
    const int& niters = 3,
    const double& percent = 0.6,
    const cv::Scalar& color = cv::Scalar(255,255,255),
    const int& pos = 0
)
{
    // Temporary Mat for intermediate processing
    cv::Mat temp;
    // check if frame is color
    if (segMask.type() == 16) cv::cvtColor(segMask, temp, cv::COLOR_BGR2GRAY);
    else segMask.copyTo(temp);

    // Vectors to store contours and hierarchy information
    std::vector< std::vector<cv::Point> > contours;
    std::vector<cv::Point> convexPts;
    std::vector<cv::Vec4i> hierarchy;

    // Apply morphological dilation to the input mask
    cv::dilate(temp, temp, cv::Mat(), cv::Point(-1, -1), niters*2);
    // Apply morphological erosion to the dilated mask
    cv::erode(temp, temp, cv::Mat(), cv::Point(-1, -1), niters);
    // Apply morphological dilation to the eroded mask
    cv::dilate(temp, temp, cv::Mat(), cv::Point(-1, -1), niters);
    cv::erode(temp, temp, cv::Mat(), cv::Point(-1, -1), 1);

    // Find contours in the processed mask
    cv::findContours(temp, contours, hierarchy, cv::RETR_CCOMP, cv::CHAIN_APPROX_SIMPLE);
    // Create an output Mat initialized with zeros
    dst = cv::Mat::zeros(segMask.size(), CV_8UC3);

    // If there are no contours, return
    if (contours.size() == 0)
        return std::tuple(cv::Point(), cv::Point(), 0.);

    // Iterate through all the top-level contours to find the largest one
    int idx = 0, largestComp = 0, scndLargest = 0;
    double maxArea = 0;
    for (; idx >= 0; idx = hierarchy[idx][0])
    {
        const std::vector<cv::Point>& c = contours[idx];

        // Calculate the area of the contour
        double area = fabs(cv::contourArea(cv::Mat(c)));

        // Update the information if the current area is larger than the maximum area
        if (area > maxArea)
        {
            maxArea = area;
            scndLargest = largestComp; 
            largestComp = idx;
        }
    }

    // CHEAT !! comment this for stability
    // if (contours.size() != 0)
    // {
    //     // use on convexity => no fit
    //     std::vector< std::vector<cv::Point> > convexList(contours.size(), std::vector<cv::Point>(contours.at(0).size()));
    //     cv::convexHull(cv::Mat(contours.at(largestComp)), convexList.at(largestComp));
    //     cv::drawContours(dst, convexList, largestComp, color, cv::FILLED, cv::LINE_8, hierarchy);
    // }
    // else cv::drawContours(dst, contours, largestComp, color, cv::FILLED, cv::LINE_8, hierarchy);
    
    // cv::fillPoly(dst, contours.at(largestComp), color, cv::LINE_8);
    cv::drawContours(dst, contours, largestComp, color, cv::FILLED, cv::LINE_8, hierarchy);
    // cv::drawContours(dst, contours, scndLargest, color, cv::FILLED, cv::LINE_8, hierarchy);
    refresh = true;

    // TODO, use cheat system to predict distance only ?
    // TODO, develop with drawContours and try the fill poly at the end
    // CHEAT !! uncomment for stability
    // PB WITH HE ORIENTATION
    // if (pos == 0 || pos == 2 || pos == 3)
    // {
    //     // cv::drawContours(dst, contours, largestComp, color, cv::FILLED, cv::LINE_8, hierarchy);
    //     // use a vector for last_big
    //     if (maxArea > (percent * last_big) )
    //     {
    //         // Draw the largest contour on the output Mat
    //         // cv::fillPoly(dst, contours.at(largestComp), color, cv::LINE_8);
    //         cv::drawContours(dst, contours, largestComp, color, cv::FILLED, cv::LINE_8, hierarchy);
    //         if (24000. > maxArea) last_big = maxArea;
    //         refresh = true;
    //     }    
    //     else 
    //     {
    //         last_mat.copyTo(dst);
    //         refresh = false;
    //     }
    // }
    // else if (pos == 4)
    // {
    //     // cv::drawContours(dst, contours, largestComp, color, cv::FILLED, cv::LINE_8, hierarchy);
    //     // use a vector for last_big
    //     if (maxArea > ((percent-0.12) * last_big) )
    //     {
    //         // Draw the largest contour on the output Mat
    //         // cv::fillPoly(dst, contours.at(largestComp), color, cv::LINE_8);
    //         cv::drawContours(dst, contours, largestComp, color, cv::FILLED, cv::LINE_8, hierarchy);
    //         // cv::drawContours(dst, contours, scndLargest, color, cv::FILLED, cv::LINE_8, hierarchy);
    //         if (24000. > maxArea) last_big = maxArea;
    //         refresh = true;
    //     }    
    //     else 
    //     {
    //         last_mat.copyTo(dst);
    //         refresh = false;
    //     }
    // }
    // else if (pos == 1)
    // {
    //     if (maxArea < ((0.6+percent) * last_big))
    //     {
    //         // cv::fillPoly(dst, contours.at(largestComp), color, cv::LINE_8);
    //         cv::drawContours(dst, contours, largestComp, color, cv::FILLED, cv::LINE_8, hierarchy);
    //         if (20000 > maxArea) last_big = maxArea;
    //         refresh = true;
    //     }
    //     else 
    //     {
    //         last_mat.copyTo(dst);
    //         refresh = false;
    //     }    
    // }
    // else 
    // {
    //     // cv::fillPoly(dst, contours.at(largestComp), color, cv::LINE_8);
    //     cv::drawContours(dst, contours, largestComp, color, cv::FILLED, cv::LINE_8, hierarchy);
    //     refresh = true;
    // }

    if (print_vectors) return getOrientation(contours[largestComp], dst);
    // TODO also return the contours to use it (refine on one hand and getorientation and distance on the other hand)
    else return getOrientation(contours[largestComp]);
}



