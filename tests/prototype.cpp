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
    bool use_alpha=false, saving=false, use_background=false;
    cv::Mat frame, fin_frame, dframe, rszd_frame, medianFrame, alpha, grayAlpha;
    const std::string window_name = "frame";
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
    auto thr_slider = 80;
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
    const int rad_puncture = 16;

    cv::Point puncture_check(430, 260);
    const int rad_check = 32;

    /// ------------------------------------

    // Define the capture and input
    std::cout << "The input is " << parser.get<std::string>("input") << std::endl;
    auto cap = initVideoCap(parser.get<std::string>("input"));
    cap >> frame;
    const auto cap_maketype = frame.type(); 
    const auto cap_size = cv::Size(
        cap.get(cv::CAP_PROP_FRAME_WIDTH),
        cap.get(cv::CAP_PROP_FRAME_HEIGHT)
    );
    // set colored overlay
    const cv::Mat color_overlay(cap_size, cap_maketype , cv::Scalar(0,0,255));
    // cv::imshow("RED", color_overlay);

    // Define ROIs to split video in 4
    const cv::Size subFrSze = {cap_size.width/2, cap_size.height/2};
    cv::Rect topleftROI = cv::Rect(0, 0, subFrSze.width, subFrSze.height), 
        toprightROI = cv::Rect(subFrSze.width, 0, subFrSze.width, subFrSze.height), 
        downleftROI = cv::Rect(0, subFrSze.height, subFrSze.width, subFrSze.height), 
        downrightROI = cv::Rect(subFrSze.width, subFrSze.height, subFrSze.width, subFrSze.height);

    cv::Mat topleftFr = cv::Mat(frame, topleftROI), 
        toprightFr = cv::Mat(frame, toprightROI), 
        downleftFr = cv::Mat(frame, downleftROI), 
        downrightFr = cv::Mat(frame, downrightROI);


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
    else
        use_alpha = false;

    std::cout << "CAP MAKETYPE : " << cap_maketype << std::endl;
    std::cout << "ALPHA MAKETYPE : " << alpha.type() << std::endl;
    std::cout << "GRAY ALPHA MAKETYPE : " << grayAlpha.type() << std::endl;


    if (parser.has("background"))
    {
        // TODO check if background is video or image
        // if video
        // auto bckgd_cap = initVideoCap(parser.get<std::string>("background"));
        // bckgd_cap.open(bckgd_frame);
        // if image
        medianFrame = cv::imread(parser.get<std::string>("background"));
        lumenCorrection(medianFrame, medianFrame);
        std::cout << "BACKGROUND INFO BEFORE PREPROCESSING : " 
                  << medianFrame.size << " & "
                  << medianFrame.type() << std::endl;
        cv::resize(medianFrame, medianFrame, cap_size,  cv::INTER_LINEAR);
		cv::cvtColor(medianFrame, medianFrame, cv::COLOR_BGR2GRAY);
        std::cout << "BACKGROUND INFO AFTER PREPROCESSING : " 
            << medianFrame.size << " & "
            << medianFrame.type() << std::endl;
        use_background = true;

        if (use_alpha)
            cv::multiply(medianFrame, grayAlpha, medianFrame);
    }
    else
    {
        if (parser.has("algo"))
        {
            // TODO check other algos or add to diff
            std::cout << "No background frame given but algo given,\n"
                      << "so initiating background subtractor" << std::endl;
            std::cout << "ALGO IS : " << parser.get<std::string>("algo") << "\n"
            << "HENCE WE HAVE THE CASE " << algos.at(parser.get<std::string>("algo")) << std::endl;
            switch (algos.at(parser.get<std::string>("algo")))
            {
            case 0:
                pBackSub = cv::createBackgroundSubtractorKNN();
                break;
            case 1:
                pBackSub = cv::createBackgroundSubtractorMOG2();
                std::cout << "BACKGROUND SUB INFO : OK " << std::endl;
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
            use_background = false;
        }
        else // DEFAULT PATH
        {
            // medianFrame = computeMedianFrame(cap, frame, false, true, 30);
            cap.set(cv::CAP_PROP_POS_FRAMES, 180);
            cap >> medianFrame;
            lumenCorrection(medianFrame, medianFrame);
            cap.set(cv::CAP_PROP_POS_FRAMES, 260);
            cv::cvtColor(medianFrame, medianFrame, cv::COLOR_BGR2GRAY);
            use_background = true;
            if (use_alpha)
            {
                std::cout << "size of medianFrame " << medianFrame.size << std::endl;
                std::cout << "size of alpha " << alpha.size << std::endl;
                cv::multiply(medianFrame, grayAlpha, medianFrame);
                std::cout << "MEDIAN MAKETYPE : " << medianFrame.type() << std::endl;
                cv::resize(medianFrame, rszd_frame, cv::Size(640, 480), cv::INTER_LINEAR);
                // cv::imshow("median", rszd_frame);
            }
        }
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
            cap_size,
            true // isColor
        );

        if (!writer.isOpened()) 
        {
            std::cerr << "INFO : Could not open the output video file for write\n";
            const bool saving = false;
        }
        const bool saving = true;
    }
    else
        const bool saving = false;
    // TODO check how to put maxFrames in the for loop
    // if (saving)
    //     const int maxFrames = videoMaxFrame(writer);

    /// ------------------------------------

    cv::Mat lab_frame;
    // TODO fix use of run time bool => find better way !!
    // loop over all frames
    for (;;)
    {
        // read frames 
        cap >> frame;
        lumenCorrection(frame, frame);
        // cv::resize(lab_frame, lab_frame, cv::Size(640, 480), cv::INTER_LINEAR);
        frame.copyTo(fin_frame);
        // Convert current frame to grayscale
		cv::cvtColor(frame, frame, cv::COLOR_BGR2GRAY);

        // TODO check this later
        // // do smthg if needs empty
        // if (frame.empty())
        // {
        //     // do things 
        //     continue;
        // }



        // std::cout << " Frame infos : " << frame.size << frame.type() << std::endl;
        if (use_alpha)
            cv::multiply(frame, grayAlpha, frame);
		
        if (use_background)
        {
            // Calculate absolute difference of current frame and the median frame
            absdiff(frame, medianFrame, dframe);

            // Threshold to binarize (will change the shadow)
            cv::threshold(dframe, dframe, thr_slider, max_threshold, cv::THRESH_BINARY);
            // on_trackbar(thr_slider)

            // Change color of the mask (might look at smthg else)
            cv::cvtColor(dframe, dframe, cv::COLOR_GRAY2BGR);
            // cv::imshow("dframe colored", dframe);
        }
        else
        {
            // std::cout << "APLYING THE SUBTRACTION " << std::endl;
            //update the background model wirth substraction
            pBackSub->apply(frame, dframe, 0.5);

            // some other algos
            // cv::grabCut()
            cv::cvtColor(dframe, dframe, cv::COLOR_GRAY2BGR);
        }

        // overlay the result over the source video
        // cv::cvtColor(frame, frame, cv::COLOR_GRAY2BGR); // just copy before ? !!!
        /// TROP LENTS !!!!!!
        cv::addWeighted(fin_frame, 1, dframe, rem_slider, 0.0, fin_frame);
        cv::multiply(dframe, color_overlay, dframe);
        cv::circle(fin_frame, puncture_zone, rad_puncture, cv::Scalar(0,255,0), 2);
        cv::circle(fin_frame, puncture_check, rad_check, cv::Scalar(0,255,0), 2);
        cv::addWeighted(fin_frame, bkcgnd_slider, dframe, overlay_slider, 0.0, fin_frame);

		// // Display Image
        // cv::resize(frame, rszd_frame, cv::Size(640, 480), cv::INTER_LINEAR);
        // cv::imshow("Test", rszd_frame);
        cv::resize(fin_frame, rszd_frame, cv::Size(640, 480), cv::INTER_LINEAR);
		if (! parser.has("hide"))
        {
            // cv::imshow("lab", lab_frame);
            cv::imshow(window_name, rszd_frame);
        }
        if(saving)
            writer.write(fin_frame);

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