#include <string>
#include <iostream>
#include <random>
#include <map>

#include "../include/image_processing/utility/constante.hpp"

#include <opencv2/core.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/video.hpp>
#include <opencv2/video/background_segm.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>



/// --------------------- Function definitions ---------------------

/// ==== New functions ====
// TODO overload functions for vectorization because of splitting in 4
// multi thread within function and synced ?
// check multithreading for android later
// (we will have to discuss the archi of app, if it will get the 4 cap or just 1)

void lumenCorrection(
    cv::Mat& src_frame,
    cv::Mat& out_frame, 
    const bool& isColor = true
)
{
    // Init variables
    cv::Mat tmp_frame;
    int f_transf, l_transf;
    
    // TODO check other cases
    if (isColor)
        f_transf = cv::COLOR_BGR2Lab, l_transf = cv::COLOR_Lab2BGR;
    else 
        std::cerr << "Frame needs to be in BGR" << std::endl;
    
    // READ RGB color image and convert it to Lab
    cv::cvtColor(src_frame, tmp_frame, f_transf);

    // Extract the L channel
    std::vector<cv::Mat> lab_planes(3);
    cv::split(tmp_frame, lab_planes);  // now we have the L image in lab_planes[0]

    // apply the CLAHE algorithm to the L channel
    cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE();
    clahe->setClipLimit(4);
    cv::Mat dst;
    clahe->apply(lab_planes[0], tmp_frame);

    // Merge the the color planes back into an Lab image
    tmp_frame.copyTo(lab_planes[0]);
    cv::merge(lab_planes, tmp_frame);
    cv::cvtColor(tmp_frame, out_frame, l_transf);
}


cv::Mat initDenoised(
    const std::string& input,
    const double& threshold = 45., // test other values
    const bool& turnGray = true
)
{
    cv::Mat frame = cv::imread(input);

    if (turnGray) cv::cvtColor(frame, frame, cv::COLOR_BGR2GRAY);
    cv::GaussianBlur(frame, frame, cv::Size(5,5),0 );
    cv::threshold(frame, frame, threshold, 255, cv::THRESH_BINARY);
    cv::erode(frame,frame, cv::Mat(), cv::Point(-1,-1), 2);
    cv::dilate(frame, frame, cv::Mat(), cv::Point(-1, -1), 2);

    return frame;
}

void initDenoised(
    cv::Mat& src_frame, 
    cv::Mat& out_frame,
    const double& threshold = 45., // test other values
    const int& niters = 2,
    const bool& turnGray = true
)
{
    cv::Mat tmp(src_frame);
    if (turnGray) cv::cvtColor(tmp, tmp, cv::COLOR_BGR2GRAY);
    cv::GaussianBlur(tmp, tmp, cv::Size(5,5),0 );
    cv::threshold(tmp, tmp, threshold, 255, cv::THRESH_BINARY);
    cv::erode(tmp,tmp, cv::Mat(), cv::Point(-1,-1), niters);
    cv::dilate(tmp, out_frame, cv::Mat(), cv::Point(-1, -1), niters);
}

void refineSegmentation(
    cv::Mat& segMask, 
    cv::Mat& dst, 
    double& last_big,
    const cv::Mat& last_mat,
    bool& refresh,
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
    std::vector<std::vector<cv::Point> > contours;
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
        return;

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
    // if (show) 
    // {
    //     std::cout << "MAX AREA : " << maxArea << std::endl;
    //     std::cout << "LAST BIG AREA : " << last_big << std::endl;

    // }
    if (pos == 0 || pos == 2 || pos == 3)
    {
        // cv::drawContours(dst, contours, largestComp, color, cv::FILLED, cv::LINE_8, hierarchy);
        // use a vector for last_big
        if (maxArea > (percent * last_big) )
        {
            // Draw the largest contour on the output Mat
            cv::drawContours(dst, contours, largestComp, color, cv::FILLED, cv::LINE_8, hierarchy);
            // cv::drawContours(dst, contours, scndLargest, color, cv::FILLED, cv::LINE_8, hierarchy);
            if (24000. > maxArea) last_big = maxArea;
            refresh = true;
        }    
        else 
        {
            last_mat.copyTo(dst);
            refresh = false;
        }
    }
    else if (pos == 4)
    {
        // cv::drawContours(dst, contours, largestComp, color, cv::FILLED, cv::LINE_8, hierarchy);
        // use a vector for last_big
        if (maxArea > ((percent-0.12) * last_big) )
        {
            // Draw the largest contour on the output Mat
            cv::drawContours(dst, contours, largestComp, color, cv::FILLED, cv::LINE_8, hierarchy);
            // cv::drawContours(dst, contours, scndLargest, color, cv::FILLED, cv::LINE_8, hierarchy);
            if (24000. > maxArea) last_big = maxArea;
            refresh = true;
        }    
        else 
        {
            last_mat.copyTo(dst);
            refresh = false;
        }
    }
    else if (pos == 1)
    {
        if (maxArea < ((0.6+percent) * last_big))
        {
            cv::drawContours(dst, contours, largestComp, color, cv::FILLED, cv::LINE_8, hierarchy);
            if (20000 > maxArea) last_big = maxArea;
            refresh = true;
        }
        else 
        {
            last_mat.copyTo(dst);
            refresh = false;
        }    
    }
    else 
    {
        cv::drawContours(dst, contours, largestComp, color, cv::FILLED, cv::LINE_8, hierarchy);
        refresh = true;
    }
}

// need to be gray
void drawHoughLines(
    cv::Mat& in_frame,
    cv::Mat& out_frame,
    const bool& standard = false
)
{
    cv::Mat tmp;
    std::vector<cv::Vec2f> lines; // will hold the results of the detection

    if (in_frame.type() == 16) cv::cvtColor(in_frame, tmp, cv::COLOR_BGR2GRAY);
    else in_frame.copyTo(tmp);

    // Edge detection
    cv::Canny(tmp, tmp, 50, 200, 3);

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

void contouring(
    cv::Mat& in_frame,
    cv::Mat& out_frame,
    const cv::Scalar& color = cv::Scalar(255,255,255)

)
{
    cv::Mat temp;
    // Vectors to store contours and hierarchy information
    std::vector<std::vector<cv::Point> > contours;
    std::vector<cv::Vec4i> hierarchy;

    // Find contours in the processed mask
    cv::findContours(in_frame, contours, hierarchy, cv::RETR_CCOMP, cv::CHAIN_APPROX_SIMPLE);

    // Create an output Mat initialized with zeros
    temp = cv::Mat::zeros(in_frame.size(), CV_8UC3);

    // If there are no contours, return
    if (contours.size() == 0)
        return;

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

    // Draw the largest contour on the output Mat
    cv::drawContours(out_frame, contours, largestComp, color, cv::FILLED, cv::LINE_8, hierarchy);

}

// void biggestContour(
//     cv::Mat& src_frame, // must go through initDenoised first
//     cv::Mat& out_frame
// )
// {
//     std::vector<std::vector<cv::Point> > contours;
//     cv::findContours(src_frame, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
//     // imutils::grabContour();
//     cv::drawContours(out_frame, contours, largestComp, color, cv::FILLED, cv::LINE_8, hierarchy);

// }

void splitTo4(
    const cv::Mat& src_frame,
    const std::vector<cv::Rect>& rectangles,
    std::vector<cv::Mat>& splitted_frames
)
{
    assert(rectangles.size() == splitted_frames.size());
    for (int pos=0; pos<rectangles.size(); pos++) 
        splitted_frames[pos] = src_frame(rectangles[pos]);
        // cv::Mat(src_frame, rectangles[pos]);
}

// TODO check the function, needs debug
void imageBipIt(
    const cv::Mat& extractFrame,
    cv::Mat& out_frame,
    const cv::Rect& ROI,
    const cv::Scalar& color
)
{
    cv::Mat checkingFrame = extractFrame(ROI);
    cv::MatIterator_<cv::Vec3b> it; // = src_it.begin<cv::Vec3b>();
    for (it = checkingFrame.begin<cv::Vec3b>(); it != checkingFrame.end<cv::Vec3b>(); ++it)
    {
        if (! ((*it)[2]==0))
        {
            cv::rectangle(out_frame, ROI, color, 3);
            break;
        }
        else cv::rectangle(out_frame, ROI, cv::Scalar(0,0,0), 3);

    }
}

// Define a pixel
typedef cv::Point_<uint8_t> Pixel;
cv::Point_<uint8_t> pt;

// struct ImageBip
// {
//     void Bipping(Pixel& pixel, const int* position) const 
//     {
//         // perform the check with the pixel
//         if (pixel[2]==0)

//     }
// }

// takes the non zero pixels from a frame (bound to a masked zone) 
// and bips if they are inside the box
void maskBip(
    const cv::Mat& mask,
    cv::Mat& plot_frame,
    const cv::Rect& ROI,
    const cv::Rect& bippinBox,
    const cv::Scalar& color = cv::Scalar(0,255,0),
    const bool& alreadyGray = false
)
{
    cv::Mat locations, tmp;
    if ( !(alreadyGray)) cv::cvtColor(mask, tmp, cv::COLOR_BGR2GRAY);
    cv::findNonZero(tmp, locations);
    cv::MatIterator_<cv::Point> it;
    for ( it = locations.begin<cv::Point>(); it != locations.end<cv::Point>(); ++it )
    {
        if ( (*it).inside(ROI) )
        {
            cv::rectangle(plot_frame, bippinBox, color, 3);
            break;
        }
        else cv::rectangle(plot_frame, bippinBox, cv::Scalar(0,0,0), 3);
    }
}

/// -------------------------
/// ==== Lib functions ====
// TODO paste these to the src because changes 

/// Device init functions

cv::VideoCapture initVideoCap(const std::string& input)
{
    cv::VideoCapture cap;
    // Determine the input source (camera or video file)
    if (input.empty())
        cap.open(0);  // If no input is provided, open the default camera (index 0)
    else if (input.size() == 1)
        cap.open(std::stoi(input));  // If the input is one string long, it is considered a number, opens the input index cam
    else
        cap.open(cv::samples::findFileOrKeep(input));  // Open the specified video file
	if(!cap.isOpened())
		std::cerr << "INFO : Error opening video file\n";
    return cap;
}

cv::VideoCapture initVideoCap(const int& input=0)
{
    cv::VideoCapture cap;
    // Determine the input source (camera or video file)
    cap.open(input);
	if(!cap.isOpened())
		std::cerr << "INFO : Error opening video file\n";
    return cap;
}

int videoMaxFrame(
    const cv::VideoCapture& cap
)
{
    if (CV_VERSION_MAJOR >= 3)
    {
        // can use smrt or shrd ptr for optim
        return cst::DBL_INFINITE;
    }
    // Get the video info
    const int fps = cap.get(cv::CAP_PROP_FPS);
    const int bitrate = cap.get(cv::CAP_PROP_BITRATE);
    // 1Gbytes = 1073741824bits
    // TODO check if cast is ok
    return std::div( 2147483648, bitrate).quot * fps; // check if pb of overflow
}

int videoMaxFrame(
    const cv::VideoWriter& writer
)
{
    if (CV_VERSION_MAJOR >= 3)
    {
        // can use smrt or shrd ptr for optim
        return cst::DBL_INFINITE;
    }
    // Get the video info
    const int fps = writer.get(cv::CAP_PROP_FPS);
    const int bitrate = writer.get(cv::CAP_PROP_BITRATE);
    // 1Gbytes = 1073741824bits
    // TODO check if cast is ok
    return std::div(2147483648, bitrate).quot * fps;
}

/// -------------------------
/// Median computation functions 

// // can be an int or a str
// template <typename CAP_INPUT>
// // can be anything
// template <typename MEDIAN>

template <typename MEDIAN> 
MEDIAN getMedian(std::vector<MEDIAN> elements) {
	nth_element(
        elements.begin(), 
        elements.begin()+elements.size()/2, 
        elements.end()
    );
	//sort(elements.begin(),elements.end());
	return elements[elements.size()/2];
}

cv::Mat compute_median(std::vector<cv::Mat> vec) {
	// Note: Expects the image to be CV_8UC3
	cv::Mat medianImg(
        vec[0].rows, 
        vec[0].cols, 
        CV_8UC3, 
        cv::Scalar(0, 0, 0)
    );

	for(int row=0; row<vec[0].rows; row++) {
		for(int col=0; col<vec[0].cols; col++) {
			std::vector<int> elements_B;
			std::vector<int> elements_G;
			std::vector<int> elements_R;

			for(int imgNumber=0; imgNumber<vec.size(); imgNumber++) {	
				int B = vec[imgNumber].at<cv::Vec3b>(row, col)[0];
				int G = vec[imgNumber].at<cv::Vec3b>(row, col)[1];
				int R = vec[imgNumber].at<cv::Vec3b>(row, col)[2];
				
				elements_B.push_back(B);
				elements_G.push_back(G);
				elements_R.push_back(R);
			}

			medianImg.at<cv::Vec3b>(row, col)[0] = getMedian(elements_B);
			medianImg.at<cv::Vec3b>(row, col)[1] = getMedian(elements_G);
			medianImg.at<cv::Vec3b>(row, col)[2] = getMedian(elements_R);
		}
	}
	return medianImg;
}

cv::Mat computeMedianFrame(
	cv::VideoCapture& cap,
	cv::Mat& frame,
    const bool& stream = false, // TODO : find a way to tell difference between file and streaming
	const bool& working=true,
    int nFrames = 25
)
{
	// Set the parameters
    nFrames = std::min(nFrames, (int) cap.get(cv::CAP_PROP_FRAME_COUNT));
	std::vector<cv::Mat> frames;

	// Randomly select nFrames frames
	std::default_random_engine generator;
	std::uniform_int_distribution<int> distribution(0, cap.get(cv::CAP_PROP_FRAME_COUNT));

	for(int i=0; i<nFrames; i++) {
		int fid = distribution(generator);
		cap.set(cv::CAP_PROP_POS_FRAMES, fid);
		cap >> frame;
		if(frame.empty())
			continue;
		frames.push_back(frame);
	}

    // set back to first frame
    cap.set(cv::CAP_PROP_POS_FRAMES, 0);
	auto medianFrame = compute_median(frames);
	// Calculate the median along the time axis
	if (working)
		cv::cvtColor(medianFrame, medianFrame, cv::COLOR_BGR2GRAY);
    return medianFrame;
}


/// -------------------------
/// Overlay computation functions

void colorBlending(
    const cv::Mat& background,
    const cv::Mat& alpha, 
    cv::Mat& outImage, 
    const cv::Scalar& BGR = clrs::RED,
    const double& cover = 0.25
)
{
    // Check if needs to convert the images to 3 channels of 32 bit float
    // foreground.convertTo(foreground, CV_32FC3);
    // background.convertTo(background, CV_32FC3);
    const cv::Size matSize= alpha.size();
    cv::Mat foreground(matSize.height, matSize.width, CV_8UC3, BGR);

    // Find the total number of pixels in the images (assuming 2D matrices)
    int numberOfPixels = foreground.rows * foreground.cols * foreground.channels();

    // Get floating point pointers to the data matrices
    double* fptr = reinterpret_cast<double*>(foreground.data);
    double* bptr = reinterpret_cast<double*>(background.data);
    double* aptr = reinterpret_cast<double*>(alpha.data);
    double* outImagePtr = reinterpret_cast<double*>(outImage.data);

    // Loop over all pixels ONCE (can be parallelized)
    for (int i = 0; i < numberOfPixels; i++, outImagePtr++, fptr++, aptr++, bptr++)
    {
        // Perform alpha blending equation: result = (foreground * cover*alpha) + (background * (1-cover - alpha))
        *outImagePtr = (*fptr) * cover*(*aptr) + (*bptr) * (1. - cover - *aptr);
    }
}

void alphaBlending(
    const cv::Mat& foreground, 
    const cv::Mat& background, 
    const cv::Mat& alpha, 
    cv::Mat& outImage
)
{
    // Check if needs to convert the images to 3 channels of 32 bit float
    // foreground.convertTo(foreground, CV_32FC3);
    // background.convertTo(background, CV_32FC3);
    
    // Find the total number of pixels in the images (assuming 2D matrices)
    int numberOfPixels = foreground.rows * foreground.cols * foreground.channels();

    // Get floating point pointers to the data matrices
    float* fptr = reinterpret_cast<float*>(foreground.data);
    float* bptr = reinterpret_cast<float*>(background.data);
    float* aptr = reinterpret_cast<float*>(alpha.data);
    float* outImagePtr = reinterpret_cast<float*>(outImage.data);

    // Loop over all pixels ONCE (can be parallelized)
    for (int i = 0; i < numberOfPixels; i++, outImagePtr++, fptr++, aptr++, bptr++)
    {
        // Perform alpha blending equation: result = (foreground * alpha) + (background * (1 - alpha))
        *outImagePtr = (*fptr) * (*aptr) + (*bptr) * (1 - *aptr);
    }
}

cv::Mat initAlphaFrame(const std::string& alphaStr)
{
    // Read the alpha frame
    auto alpha = cv::imread(alphaStr);
    // Normalize the alpha mask to keep intensity between 0 and 1
    alpha.convertTo(alpha, CV_32FC3, 1.0/255);
    return alpha;
}

cv::Mat initAlphaFrame(
    const std::string& alphaStr, const cv::Size& frameResize
)
{
    // Read the alpha frame
    auto alpha = cv::imread(alphaStr);
    // Resize the alpha frame
    resize(alpha, alpha, frameResize, cv::INTER_LINEAR);
    // Normalize the alpha mask to keep intensity between 0 and 1
    alpha.convertTo(alpha, CV_32FC3, 1.0/255);
    return alpha;
}

cv::Mat initAlphaFrame(
    const std::string& alphaStr, 
    const cv::Size& frameResize,
    const int& maketype
)
{
    // Read the alpha frame
    auto alpha = cv::imread(alphaStr);
    // Resize the alpha frame
    resize(alpha, alpha, frameResize, cv::INTER_LINEAR);
    // Normalize the alpha mask to keep intensity between 0 and 1
    alpha.convertTo(alpha, maketype, 1.0/255);
    return alpha;
}

/// =============================================================
/// --------------------- Region definition ---------------------
/// =============================================================

// #define lumen
// #define predenoising
#define postdenoising
#define refining
#define showIntermediate
#define checkingBox

/// ================================================
/// --------------------- Main ---------------------
/// ================================================

const std::string params
    = "{ help h       |        | Print usage }"
      "{ input i      |        | Source video to compute, 2sec of background at beginning }" 
      "{ alpha a      | <none> | Alpha mask to determine which part must be treated }"
      "{ background b | <none> | Background that can be used as median frame instead of computing it }" 
      "{ algo c       | <none> | Algo to use if there is no alpha mask given}"
      "{ output o     | <none> | Output file where to save the result of computation }" 
      "{ hide h       | <none> | If non null hides the video}"
      ;

int main(int argc, char const *argv[])
{
    /// ------------------------------------

    // Parse the args
    cv::CommandLineParser parser(argc, argv, params);
    parser.about(
        "This program is a prototype of how to segment element"
        "based on the comparaison with the background\n"
    );
    if (parser.has("help") || argc <= 1)
    {
        //print help information
        parser.printMessage();
    }

    /// ------------------------------------

    // set the parameters to use (check to init correctly)
    bool use_alpha=false, saving=false, use_background=false, use_algo=false, 
    hide=false, punctured=false, refresh=true, scnd_step=false, post_puncture_stop = false;
    cv::Mat frame, fin_frame, dframe, rszd_frame, medianFrame, alpha, 
    grayAlpha, bckg_frame, algo_frame, refine_frame, stopping_frame;
    const std::string window_name = "frame";
    double percent = 0.85 ;
    cv::VideoWriter writer;
    cv::namedWindow(window_name); // create window
    
    //create Background Subtractor objects
    cv::Ptr<cv::BackgroundSubtractor> pBackSub;

    // ------ TODO add a taskbar to change threshold and test ------

    // // function for threshold trackbar callback
    // void thr_trackbar(int, void*)
    // {
    //     threshold(dframe, dframe, thr_slider, 255, cv::THRESH_BINARY);
    // }
    // // create taskbar for the diff threshold
    const auto max_threshold = 255.;
    auto thr_slider = 100;
    // char*  trackbar_name = "diff threshold"; // check if possible to add const
    // sprintf(trackbar_name, " threshold x %d", max_threshold);
    // cv::createTrackbar(
    //     trackbar_name, 
    //     window_name, 
    //     &thr_slider, 
    //     max_threshold, 
    //     thr_trackbar
    // );

    // // create the taskbar for the overlay power
    const auto max_overlay = 255.;
    auto overlay_slider = 1.;
    auto bkcgnd_slider = 0.8;
    auto rem_slider = -0.8;
    // char* overlay_trackbar_name = "overlay threshold";
    // sprintf(overlay_trackbar_name, "overlay x %d", max_overlay);
    // cv::createTrackbar(  
    //     overlay_trackbar_name, 
    //     window_name,
    //     &overlay_slider,
    //     max_overlay,
    //     cv::addWeighted
    // );

    const std::map<std::string, int> algos = 
    {
        {"KNN", 0},
        {"MOG2", 1},
        {"CNT", 2},
        {"GMG", 3},
        {"GSOC", 4},
        {"LSBP", 5},
        {"MOG", 6},
        {"cudaFGD", 7},
        {"cudaGMG", 8},
        {"cudaMOG", 9},
    };

    // insert point to puncture
    cv::Point puncture_zone(430, 260);
    const int rad_puncture = 24;

    // init of the validation frame after init of frame and video capture
    cv::Point puncture_check(430, 260);
    const int rad_check = 48;

    /// ------------------------------------

    // Define the capture and input
    auto cap = initVideoCap(parser.get<std::string>("input"));
    cap >> frame;
    const auto cap_maketype = frame.type(); 
    const auto cap_size = cv::Size(
        cap.get(cv::CAP_PROP_FRAME_WIDTH),
        cap.get(cv::CAP_PROP_FRAME_HEIGHT)
    );
    // set colored overlay
    const cv::Mat color_overlay(cap_size, cap_maketype , cv::Scalar(0,0,255));

    // Define ROIs to split video in 4 (5 actually)
    const cv::Size subFrSze = {cap_size.width/2, cap_size.height/2};
    const cv::Rect topleftROI = cv::Rect(0, 0, subFrSze.width, subFrSze.height), 
        toprightupROI = cv::Rect(subFrSze.width, 0, subFrSze.width, 0.5*subFrSze.height), 
        toprightdownROI = cv::Rect(subFrSze.width, 0.5*subFrSze.height, subFrSze.width, 0.5*subFrSze.height), 
        downleftROI = cv::Rect(0, subFrSze.height, subFrSze.width, subFrSze.height), 
        downrightROI = cv::Rect(subFrSze.width, subFrSze.height, subFrSze.width, subFrSze.height);
    
    const std::map<int, cv::Rect> rois = 
    {
        {0, topleftROI},
        {1, downleftROI},
        {2, toprightdownROI},
        {3, toprightupROI},
        {4, downrightROI},
    };

    // init value to prevent clipping
    std::vector<double> last_big = {0.,1000000.,0.,0.,0.};

    cv::Mat topleftFr = cv::Mat(frame, topleftROI), 
        toprightupFr = cv::Mat(frame, toprightupROI),
        toprightdownFr = cv::Mat(frame, toprightdownROI),  
        downleftFr = cv::Mat(frame, downleftROI), 
        downrightFr = cv::Mat(frame, downrightROI);

    const std::vector<cv::Rect> rectangles ({topleftROI, downleftROI, toprightdownROI, toprightupROI, downrightROI});
    std::vector<cv::Mat> splitted(rois.size()), dvector(rois.size()), last_vector(rois.size()), fin_vector(rois.size()); // init splitted vector and transformed vector
    int minRoiToProcess=0, roiToProcess=rois.size() ;
    splitTo4(frame, rectangles, splitted);

    const cv::Rect validation = cv::Rect(
        puncture_zone.x - rad_puncture, 
        puncture_zone.y - rad_puncture,
        2*rad_puncture,
        2*rad_puncture
    );
    cv::Mat validationFrame = frame(validation);

    const cv::Rect caution = cv::Rect(
        1.7 * subFrSze.width,
        1.1*subFrSze.height,
        0.05*subFrSze.width,
        0.4*subFrSze.height
    );
    cv::Mat warningFrame = frame(caution);

    const cv::Rect forbiden = cv::Rect(
        1.75 * subFrSze.width,
        1.05*subFrSze.height,
        0.25 * subFrSze.width,
        0.5*subFrSze.height
    );
    cv::Mat forbidenFrame = frame(forbiden);

    // used for the rectangle bipping
    cv::MatIterator_<cv::Vec3b> it, otherIt; // = src_it.begin<cv::Vec3b>();

    // Define way to get element, background priority
    if (parser.has("alpha"))
    {
        // Init alpha frames
        alpha = initAlphaFrame(
            parser.get<std::string>("alpha"),
            cap_size,
            cap_maketype
        );
        cv::cvtColor(alpha, grayAlpha, cv::COLOR_BGR2GRAY);
        use_alpha = true;
    }

    if (parser.has("background"))
    {
        medianFrame = cv::imread(parser.get<std::string>("background"));
        #ifdef lumen
        lumenCorrection(medianFrame, medianFrame);
        #endif
       
        cv::resize(medianFrame, medianFrame, cap_size,  cv::INTER_LINEAR);
		cv::cvtColor(medianFrame, medianFrame, cv::COLOR_BGR2GRAY);

        if (use_alpha)
            cv::multiply(medianFrame, grayAlpha, medianFrame);

        // check if it can be colored
        // cv::cvtColor(medianFrame, medianFrame, cv::COLOR_GRAY2BGR);

        use_background = true;
    }
    else 
    {

    }

    if (parser.has("algo"))
    {
        // TODO check other algos or add to diff
        switch (algos.at(parser.get<std::string>("algo")))
        {
        case 0:
            pBackSub = cv::createBackgroundSubtractorKNN(700, 375, false);
            break;
        case 1:
            pBackSub = cv::createBackgroundSubtractorMOG2(1000, 200., false);
            break;
        // case 2:
        //     pBackSub = cv::bgsegm::createBackgroundSubtractorCNT();
        //     break;
        // case 3:
        // when using GMG use pmorphologyEx to denoise(good edges)
        //     pBackSub = cv::bgsegm::createBackgroundSubtractorGMG();
        //     break;
        // case 4:
        //     pBackSub = cv::bgsegm::createBackgroundSubtractorMGSOC();
        //     break;
        // case 5:
        //     pBackSub = cv::bgsegm::createBackgroundSubtractorLSBP();
        //     break;
        // case 6:
        //     pBackSub = cv::bgsegm::createBackgroundSubtractorMOG();
        //     break;
        // case 7:
        //     pBackSub = cv::cuda::createBackgroundSubtractorFGD();
        //     break;
        // case 8:
        //     pBackSub = cv::cuda::createBackgroundSubtractorGMG();
        //     break;
        // case 9:
        //     pBackSub = cv::cuda::createBackgroundSubtractorMOG();
        //     break;
        default:
            std::cerr << "INFO : No algo correct algo was given, you can choose among :\n"
                        << "MOG2, KNN,  / \n"
                        << "don't work -> CNT,GMG, GSOC, SBP, MOG, cudaFGD, cudaGMG, cudaMOG" << std::endl;
            return -1;
        }
        use_algo = true;
    }


    // Define the output where to save result
    if (parser.has("output"))
    {
        std::cout << " WRITER INFO " 
        << parser.get<std::string>("output")  << " & "
        << cv::VideoWriter::fourcc('M','J','P','G') << " & "
        << cap.get(cv::CAP_PROP_XI_FRAMERATE) << " & "
        << cap_size << std::endl; 
        writer.open(
            parser.get<std::string>("output"),
            cv::VideoWriter::fourcc('M','J','P','G'),
            30.,
            splitted[0].size(),
            true // isColor
        );

        if (!writer.isOpened()) 
            std::cerr << "INFO : Could not open the output video file for write\n";
        else saving = true;
    }

    // TODO check how to put maxFrames in the for loop
    // if (saving)
    //     const int maxFrames = videoMaxFrame(writer);

    if (parser.has("hide")) hide=true;

    /// ------------------------------------

    // prototype beggining
    cap.set(cv::CAP_PROP_POS_FRAMES, 1100);
    cap >> frame;


    cv::Mat lab_frame, defineSegm, tmp_frame;
    // TODO fix use of run time bool => find better way !!
    // loop over all frames
    for (;;)
    {
        // read frames
        cap >> frame;

        // add thresholds and denoising ?

        #ifdef lumen
        lumenCorrection(frame, frame);
        #endif

        #ifdef predenoising
        initDenoised(frame, frame, 80, 2, false);
        #endif

        frame.copyTo(fin_frame);
        // Convert current frame to grayscale
		cv::cvtColor(frame, frame, cv::COLOR_BGR2GRAY);

        // #ifdef showIntermediate
        // cv::resize(frame, rszd_frame, cv::Size(640, 480), cv::INTER_LINEAR);
        // cv::imshow("preprocessed image", rszd_frame);
        // #endif

        if (use_alpha) cv::multiply(frame, grayAlpha, frame);

        if (use_background)
        {
            // Calculate absolute difference of current frame and the median frame
            absdiff(frame, medianFrame, bckg_frame);
            // check if increasing the contrast
            // lumenCorrection(bckg_frame, bckg_frame, false);
        }

        if (use_algo)
        {
            //update the background model wirth substraction
            pBackSub->apply(frame, algo_frame, 0.12);
        }

        #ifdef showIntermediate
        cv::resize(bckg_frame, rszd_frame, cv::Size(640, 480), cv::INTER_LINEAR);
        cv::imshow("background image", rszd_frame);
        cv::resize(algo_frame, rszd_frame, cv::Size(640, 480), cv::INTER_LINEAR);
        cv::imshow("algo frame", rszd_frame);
        #endif

        #ifdef postdenoising

            // Threshold to binarize (will change the shadow)
            cv::threshold(bckg_frame, bckg_frame, 120, max_threshold, cv::THRESH_BINARY);      
            // Change color of the mask (might look at smthg else)
            cv::cvtColor(bckg_frame, bckg_frame, cv::COLOR_GRAY2BGR);

            // Threshold to binarize (will change the shadow)
            cv::threshold(algo_frame, algo_frame, 80, max_threshold, cv::THRESH_BINARY);      
            // Change color of the mask (might look at smthg else)
            cv::cvtColor(algo_frame, algo_frame, cv::COLOR_GRAY2BGR);

            // #ifdef showIntermediate
            // cv::resize(bckg_frame, rszd_frame, cv::Size(640, 480), cv::INTER_LINEAR);
            // cv::imshow("background image denoised", rszd_frame);
            // cv::resize(algo_frame, rszd_frame, cv::Size(640, 480), cv::INTER_LINEAR);
            // cv::imshow("algo frame denoised", rszd_frame);
            // #endif

        #endif

        cv::addWeighted(bckg_frame, 1., algo_frame, 1., 0., dframe);

        #ifdef showIntermediate
        // cv::circle(dframe, puncture_check, rad_check, cv::Scalar(0,255,0), 2);
        cv::resize(dframe, rszd_frame, cv::Size(640, 480), cv::INTER_LINEAR);
        cv::imshow("summed image", rszd_frame);
        #endif

        #ifdef refining
        punctured = (cap.get(cv::CAP_PROP_POS_FRAMES) > 930) && (cap.get(cv::CAP_PROP_POS_FRAMES) < 1190);
        dframe.copyTo(refine_frame);
        splitTo4(refine_frame, rectangles, dvector);
        // for(int pos=0; pos<rectangles.size(); pos++)
        for(int pos=minRoiToProcess; pos<roiToProcess; pos++)
        {
            refineSegmentation(
                dvector[pos], 
                dvector[pos], 
                last_big[pos], 
                last_vector[pos],
                refresh,
                2, percent,
                cv::Scalar(0,0,255),
                pos
            );
            // add variable for the saving condition
            // save all parts independently
            // push dvector inside of dframe
            // refreshing_state_bef_puncture = ((pos < 3) ^ (punctured));
            // if (refresh && refreshing_state_bef_puncture) 

            post_puncture_stop = !(punctured && (pos < 3));
            if (refresh && post_puncture_stop)
                dvector[pos].copyTo(last_vector[pos]);
            last_vector[pos].copyTo(dframe(rois.at(pos)));
        }
        #endif

        #ifdef checkingBox
        // dosn't work (makes the prog lag)
        // imageBip(dframe, fin_frame, topleftROI, cv::Scalar(0, 255,0));
        // std::cout << cap.get(cv::CAP_PROP_POS_FRAMES) << std::endl;

        if (cap.get(cv::CAP_PROP_POS_FRAMES) < 1190)
        {
            // validation bipping green
            // maskBip(dframe, fin_frame, validation, topleftROI,cv::Scalar(0,255,0) );
            validationFrame = dframe(validation);
            for (it = validationFrame.begin<cv::Vec3b>(); it != validationFrame.end<cv::Vec3b>(); ++it)
            {
                if (! ((*it)[2]==0))
                {
                    cv::rectangle(fin_frame, topleftROI, cv::Scalar(0,255,0), 3);
                    break;
                }
                else cv::rectangle(fin_frame, topleftROI, cv::Scalar(0,0,0), 3);
            }
        }
        else
        {
            cv::rectangle(fin_frame, topleftROI, cv::Scalar(0,255,0), 3);
            // checking distannce of needle
            // maskBip(dframe, fin_frame, caution, downrightROI,cv::Scalar(0,125,255) );
            warningFrame = dframe(caution);
            for (it = warningFrame.begin<cv::Vec3b>(); it != warningFrame.end<cv::Vec3b>(); ++it)
            {
                if (! ((*it)[2]==0))
                {
                    cv::rectangle(fin_frame, downrightROI, cv::Scalar(0,112,112), 3);
                    for (otherIt = forbidenFrame.begin<cv::Vec3b>(); 
                        otherIt != forbidenFrame.end<cv::Vec3b>(); 
                        ++otherIt
                    )
                    {
                        if (! ((*otherIt)[2]==0))
                        {
                            cv::rectangle(fin_frame, downrightROI, cv::Scalar(0,0,255), 3);
                            break;
                        }
                    }
                    break;
                }
                else cv::rectangle(fin_frame, downrightROI, cv::Scalar(0,0,0), 3);
            }

            // maskBip(dframe, fin_frame, forbiden, downrightROI,cv::Scalar(0,0,255) );
            // forbidenFrame = dframe(forbiden);
            // for (it = forbidenFrame.begin<cv::Vec3b>(); it != forbidenFrame.end<cv::Vec3b>(); ++it)
            // {
            //     if (! ((*it)[2]==0))
            //     {
            //         cv::rectangle(fin_frame, downrightROI, cv::Scalar(0,0,255), 3);
            //         break;
            //     }
            //     else cv::rectangle(fin_frame, downrightROI, cv::Scalar(0,0,0), 3);
            // }
        }
        #endif


        #ifdef showIntermediate
        cv::resize(dframe, rszd_frame, cv::Size(640, 480), cv::INTER_LINEAR);
        cv::imshow("refined image", rszd_frame);
        #endif

        // overlay the result on the source video
        cv::subtract(fin_frame, dframe, fin_frame);
        cv::multiply(dframe, color_overlay, dframe);
        cv::circle(fin_frame, puncture_zone, rad_puncture, cv::Scalar(0,255,0), 2);
        cv::addWeighted(fin_frame, bkcgnd_slider, dframe, overlay_slider, 0.0, fin_frame);

        // cv::rectangle(fin_frame, caution, cv::Scalar(0,128,255), 2);
        // cv::rectangle(fin_frame, forbiden, cv::Scalar(0,0,255), 2);
		// Display Image
        cv::resize(fin_frame, rszd_frame, cv::Size(640, 480), cv::INTER_LINEAR);

        splitTo4(fin_frame, rectangles, fin_vector);
		if (! parser.has("hide"))
        {
            // cv::imshow("lab", lab_frame);
            cv::imshow(window_name, rszd_frame);
        }
        if(saving)
            // writer.write(fin_vector[0]);
            writer.write(fin_frame(downrightROI));
        // std::cout << cap.get(cv::CAP_PROP_POS_FRAMES) << std::endl;
        
        // TODO add the handle function later
        // Exit if ESC pressed
        int key_event = cv::waitKey(1);
        if(key_event == 27) 
        {
            break;
        }
    }
    return 0;
}