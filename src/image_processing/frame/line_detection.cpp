#include <image_processing/frame/line_detection.hpp>


void drawHoughLines(
    cv::Mat& in_frame,
    cv::Mat& out_frame,
    const double& cannyThreshold1 = 50,
    const double& cannyThreshold2 = 200,
    const bool& standard = false
)
{
    cv::Mat tmp;
    std::vector<cv::Vec2f> lines; // will hold the results of the detection

    if (in_frame.type() == 16) cv::cvtColor(in_frame, tmp, cv::COLOR_BGR2GRAY);
    else in_frame.copyTo(tmp);

    // Edge detection
    cv::Canny(tmp, tmp, cannyThreshold1, cannyThreshold2, 3);

    if (standard)
    {
        // Standard Hough Line Transform
        cv::HoughLines(tmp, lines, 1, CV_PI/180, 150, 0, 0 ); // runs the actual detection

        // Draw the lines
        for( size_t i = 0; i < lines.size(); i++ )
        {
            float rho = lines[i][0], theta = lines[i][1];
            cv::Point pt1, pt2;
            double a = cos(theta), b = sin(theta);
            double x0 = a*rho, y0 = b*rho;
            pt1.x = cvRound(x0 + 1000*(-b));
            pt1.y = cvRound(y0 + 1000*(a));
            pt2.x = cvRound(x0 - 1000*(-b));
            pt2.y = cvRound(y0 - 1000*(a));
            cv::line( tmp, pt1, pt2, cv::Scalar(0,0,255), 3, cv::LINE_AA);
        }
    }
    else
    {
        // Probabilistic Line Transform
        std::vector<cv::Vec4i> linesP; // will hold the results of the detection
        cv::HoughLinesP(tmp, linesP, 1, CV_PI/180, 50, 50, 10 ); // runs the actual detection
        // Draw the lines
        for( size_t i = 0; i < linesP.size(); i++ )
        {
            cv::Vec4i l = linesP[i];
            cv::line( tmp, cv::Point(l[0], l[1]), cv::Point(l[2], l[3]), cv::Scalar(0,0,255), 3, cv::LINE_AA);
        }
    }
    tmp.copyTo(out_frame);    
}

void drawHoughLinesProbabilistic(
    cv::Mat& in_frame,
    cv::Mat& out_frame,
    const double& cannyThreshold1 = 50,
    const double& cannyThreshold2 = 200
)
{
    cv::Mat tmp;
    std::vector<cv::Vec2f> lines; // will hold the results of the detection

    if (in_frame.type() == 16) cv::cvtColor(in_frame, tmp, cv::COLOR_BGR2GRAY);
    else in_frame.copyTo(tmp);

    // Edge detection
    cv::Canny(tmp, tmp, cannyThreshold1, cannyThreshold2, 3);

    // Probabilistic Line Transform
    std::vector<cv::Vec4i> linesP; // will hold the results of the detection
    cv::HoughLinesP(tmp, linesP, 1, CV_PI/180, 50, 50, 10 ); // runs the actual detection
    // Draw the lines
    for( size_t i = 0; i < linesP.size(); i++ )
    {
        cv::Vec4i l = linesP[i];
        cv::line( tmp, cv::Point(l[0], l[1]), cv::Point(l[2], l[3]), cv::Scalar(0,0,255), 3, cv::LINE_AA);
    }

    tmp.copyTo(out_frame);
}

void drawHoughLinesStandard(
    cv::Mat& in_frame,
    cv::Mat& out_frame,
    const double& cannyThreshold1 = 50,
    const double& cannyThreshold2 = 200
)
{
    cv::Mat tmp;
    std::vector<cv::Vec2f> lines; // will hold the results of the detection

    if (in_frame.type() == 16) cv::cvtColor(in_frame, tmp, cv::COLOR_BGR2GRAY);
    else in_frame.copyTo(tmp);

    // Edge detection
    cv::Canny(tmp, tmp, cannyThreshold1, cannyThreshold2, 3);

    // Standard Hough Line Transform
    cv::HoughLines(tmp, lines, 1, CV_PI/180, 150, 0, 0 ); // runs the actual detection

    // Draw the lines
    for( size_t i = 0; i < lines.size(); i++ )
    {
        float rho = lines[i][0], theta = lines[i][1];
        cv::Point pt1, pt2;
        double a = cos(theta), b = sin(theta);
        double x0 = a*rho, y0 = b*rho;
        pt1.x = cvRound(x0 + 1000*(-b));
        pt1.y = cvRound(y0 + 1000*(a));
        pt2.x = cvRound(x0 - 1000*(-b));
        pt2.y = cvRound(y0 - 1000*(a));
        cv::line( tmp, pt1, pt2, cv::Scalar(0,0,255), 3, cv::LINE_AA);
    }    

    tmp.copyTo(out_frame);    
}