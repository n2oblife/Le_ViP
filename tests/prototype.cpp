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



/// -------------------------
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
		std::cerr << "Error opening video file\n";
        return;
    return cap;
}

cv::VideoCapture initVideoCap(const int& input=0)
{
    cv::VideoCapture cap;
    // Determine the input source (camera or video file)
    cap.open(input);
	if(!cap.isOpened())
		std::cerr << "Error opening video file\n";
        return;
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

	const auto medianFrame = compute_median(frames);
	// Calculate the median along the time axis
	if (working)
	{
		cv::Mat grayMedianFrame;
		cv::cvtColor(medianFrame, grayMedianFrame, cv::COLOR_BGR2GRAY);
		return grayMedianFrame;
	}
	else
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
    float* fptr = reinterpret_cast<float*>(foreground.data);
    float* bptr = reinterpret_cast<float*>(background.data);
    float* aptr = reinterpret_cast<float*>(alpha.data);
    float* outImagePtr = reinterpret_cast<float*>(outImage.data);

    // Loop over all pixels ONCE (can be parallelized)
    for (int i = 0; i < numberOfPixels; i++, outImagePtr++, fptr++, aptr++, bptr++)
    {
        // Perform alpha blending equation: result = (foreground * cover*alpha) + (background * (1-cover - alpha))
        *outImagePtr = (*fptr) * cover*(*aptr) + (*bptr) * (1 - cover - *aptr);
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

/// ================================================
/// --------------------- Main ---------------------
/// ================================================

const std::string params
    = "{ help h       |        | Print usage }"
      "{ source s     |        | Source video to compute, 2sec of background at beginning }" 
      "{ alpha a      | <none> | Alpha mask to determine which part must be treated }"
      "{ background b | <none> | Background that can be used as median frame instead of computing it }" 
      "{ algo c       | <none> | Algo to use if there is no alpha mask given}"
      "{ output o     | <none> | Output file where to save the result of computation }" 
      ;

int main(int argc, char const *argv[])
{
    // Parse the args
    cv::CommandLineParser parser(argc, argv, params);
    parser.about(
        "This program is a prototype of how to segment element"
        "based on the comparaison with the background\n"
    );
    if (parser.has("help"))
    {
        //print help information
        parser.printMessage();
    }

    // set the parameters to use
    cv::Mat frame, dframe, rszd_frame, medianFrame;
    const cv::Mat color_overlay(cv::Scalar(0,0,255));
    const std::string window_name = "frame";
    cv::namedWindow(window_name); // create window

    // ------ TODO add a taskbar to change threshold and test ------

    // // function for threshold trackbar callback
    // void thr_trackbar(int, void*)
    // {
    //     threshold(dframe, dframe, thr_slider, 255, cv::THRESH_BINARY);
    // }
    // // create taskbar for the diff threshold
    const auto max_threshold = 255.;
    auto thr_slider = 30;
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
    auto overlay_slider = 0.3;
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

    // Define the capture and input
    std::cout << "The input is " << parser.get<std::string>("input") << std::endl;
    auto cap = initVideoCap(parser.get<std::string>("input"));

    // Define way to get element, background priority
    if (parser.has("alpha"))
    {
        // Init alpha frames
        const auto alpha = initAlphaFrame(
            parser.get<std::string>("alpha")
        );
        const bool use_alpha = true;
    }
    else
        const bool use_alpha = false;

    if (parser.has("background"))
    {
        // TODO check if background is video or image
        // if video
        // auto bckgd_cap = initVideoCap(parser.get<std::string>("background"));
        // bckgd_cap.open(bckgd_frame);
        // if image
        medianFrame = cv::imread(parser.get<std::string>("background"));
		cv::cvtColor(medianFrame, medianFrame, cv::COLOR_BGR2GRAY);
        const bool use_background = true;

        if (use_alpha)
            cv::multiply(medianFrame, alpha, medianFrame);
    }
    else
    {
        if (parser.has("algo"))
        {
            // TODO check other algos or add to diff
            std::cout << "No background frame given but algo given,\n"
                      << "so initiating background subtractor" << std::endl;

            //create Background Subtractor objects
            cv::Ptr<cv::BackgroundSubtractor> pBackSub;

            switch (algos.at(parser.get<std::string>("algo")))
            {
            case 0:
                pBackSub = cv::createBackgroundSubtractorMOG2();
                break;
            case 1:
                pBackSub = cv::createBackgroundSubtractorKNN();
                break;
            // case 2:
            //     pBackSub = cv::bgsegm::createBackgroundSubtractorCNT();
            //     break;
            // case 3:
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
                std::cerr << "No algo correct algo was given, you can choose among :\n"
                        << "MOG2, KNN / don't work -> CNT,GMG, GSOC, SBP, MOG, cudaFGD, cudaGMG, cudaMOG" << std::endl;
                return -1;
            }
            const bool use_background = false;
        }
        else // DEFAULT PATH
        {
            medianFrame = computeMedianFrame(cap, frame, true, 60);
            const bool use_background = true;
            if (use_alpha)
                cv::multiply(medianFrame, alpha, medianFrame);
        }
    }

    // Define the output where to save result
    if (parser.has("output"))
    {
        cv::VideoWriter writer = cv::VideoWriter(
            parser.get<std::string>("output"),
            cv::CAP_ANY, // default api
            cv::VideoWriter::fourcc('M','J','P','G'),
            30.,
            cv::Size(1920, 1080),
            true // isColor
        );

        if (!writer.isOpened()) 
        {
            std::cerr << "Could not open the output video file for write\n";
            const bool saving = false;
        }
        const bool saving = true;
    }
    else
        const bool saving = false;
    // TODO check how to put maxFrames in the for loop
    // if (saving)
    //     const int maxFrames = videoMaxFrame(writer);


    // TODO fix use of run time bool => find better way !!
    // loop over all frames
    for (;;)
    {
        // read frames 
        cap >> frame;

        // TODO check this later
        // // do smthg if needs empty
        // if (frame.empty())
        // {
        //     // do things 
        //     continue;
        // }


        // Convert current frame to grayscale
		cvtColor(frame, frame, cv::COLOR_BGR2GRAY);
        if (use_alpha)
            cv::multiply(frame, alpha, frame);
		
        if (use_background)
        {
            // Calculate absolute difference of current frame and the median frame
            absdiff(frame, medianFrame, dframe);

            // Threshold to binarize (will change the shadow)
            cv::threshold(dframe, dframe, thr_slider, max_threshold, cv::THRESH_BINARY);
            // on_trackbar(thr_slider)
            // cv::threshold(dframe, dframe, 30, 255, cv::THRESH_BINARY);

            // Change color of the mask (might look at smthg else)
            cv::multiply(dframe, color_overlay, dframe);
        }
        else
        {
            //update the background model
            pBackSub->apply(frame, dframe);
        }

        // overlay the result over the source video
        cv::addWeighted(frame, 1., dframe, overlay_slider, 0.0);

		// Display Image
        resize(dframe, rszd_frame, cv::Size(640, 480), cv::INTER_LINEAR);
		imshow(window_name, rszd_frame);

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