#include <string>
#include <iostream>
#include <random>

#include <opencv2/core.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>


/// --------------------- Function definitions ---------------------

/// ==== New functions ====

// function for threshold trackbar
static void on_trackbar( const int& thr_slider)
{
    threshold(dframe, dframe, thr_slider, 255, cv::THRESH_BINARY);
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
        cap.open(samples::findFileOrKeep(input));  // Open the specified video file
	if(!cap.isOpened())
		std::cerr << "Error opening video file\n";
        return break;
    return cap
}

cv::VideoCapture initVideoCap(const int& input)
{
    cv::VideoCapture cap;
    // Determine the input source (camera or video file)
    if (input.empty())
        cap.open(0);  // If no input is provided, open the default camera (index 0)
    else
        cap.open(input);
	if(!cap.isOpened())
		std::cerr << "Error opening video file\n";
        return break;
    return cap;
}

int videoMaxFrame(
    const cv::VideoCapture& cap
)
{
    if (cv::CV_VERSION_MAJOR >= 3)
    {
        // can use smrt or shrd ptr for optim
        return cst::DBL_INFINITE;
    }
    // Get the video info
    const int fps = cap.get(cv::CAP_PROP_FPS);
    const int bitrate = cap.get(cv::CAP_PROP_BITRATE);
    // 1Gbytes = 1073741824bits
    // TODO check if cast is ok
    return (int) std::div_t(2147483648, bitrate).quot * fps;
}

int videoMaxFrame(
    const cv::VideoWriter& writer
)
{
    if (cv::CV_VERSION_MAJOR >= 3)
    {
        // can use smrt or shrd ptr for optim
        return cst::DBL_INFINITE;
    }
    // Get the video info
    const int fps = writer.get(cv::CAP_PROP_FPS);
    const int bitrate = writer.get(cv::CAP_PROP_BITRATE);
    // 1Gbytes = 1073741824bits
    // TODO check if cast is ok
    return (int) std::div_t(2147483648, bitrate).quot * fps;
}

/// -------------------------
/// Median computation functions 

// can be an int or a str
template <typename CAP_INPUT>;
template <typename MEDIAN>;

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
	const cv::VideoCapture& cap,
	cv::Mat& frame,
	const bool& working=true,
    int& nFrames = 25
)
{
	// Set the parameters
    nFrames = std::min(nFrames, cap.get(cv::CAP_PROP_FRAME_COUNT));
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
    const auto& BGR = clrs::RED,
    const double& cover = 0.25
)
{
    // Check if needs to convert the images to 3 channels of 32 bit float
    // foreground.convertTo(foreground, CV_32FC3);
    // background.convertTo(background, CV_32FC3);
    cv:Size matSize= alpha.size();
    cv::Mat foreground(matSize[0], matSize[1], CV_8UC3, BGR);

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
    resize(alpha, alpha, frameResize, INTER_LINEAR);
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
    cv::Mat frame, dframe, rszd_frame, bckgd_frame, medianFrame, grayMedianFrame;
    const cv::Mat color_overlay(cv::Scalar())
    const std::string window_name = "frame";
    cv::namedWindow(window_name) // create window

    // TODO add a taskbar to change threshold and test
    // create taskbar for the diff threshold
    const double max_threshold = 255;
    double thr_slider;
    const std::string trackbar_name = "diff threshold";
    sprintf(trackbar_name, " threshold x %d", max_threshold);
    cv::createTrackbar(
        trackbar_name, 
        window_name, 
        &thr_slider, 
        max_threshold, 
        cv::threshold
    )

    // create the taskbar for the overlay power
    const int max_overlay = 255.;
    double overlay_slider;
    const std::string overlay_trackbar_name = "overlay threshold";
    sprintf(
        overlay_trackbar_name, 
        window_name,
        &overlay_slider,
        max_overlay,
        cv::addWeighted
    );

    // Define the capture and input
    std::cout << "The input is " << parser.get<std::string>("input") << std::endl;
    const auto cap = initVideoCap(parser.get<std::string>("input"));

    // Define way to get element, background priority
    if (parser.has("alpha"))
    {
        // Init alpha frames
        auto alpha = initAlphaFrame(
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
            cv::Ptr<BackgroundSubtractor> pBackSub;
            if (parser.get<std::string>("algo") == "MOG2")
                pBackSub = cv::createBackgroundSubtractorMOG2();
            else
                pBackSub = cv::createBackgroundSubtractorKNN();

            const bool use_background = false;
        }
        else
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
        )

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
		// Calculate absolute difference of current frame and the median frame
		absdiff(frame, medianFrame, dframe);

        // TODO add the taskbar to play with threshold
		// Threshold to binarize (will change the shadow)
        cv::threshold(dframe, dframe, thr_slider, max_threshold, cv::THRESH_BINARY);
        // on_trackbar(thr_slider)
		// cv::threshold(dframe, dframe, 30, 255, cv::THRESH_BINARY);

        cv::multiply(dframe, colored_frame, dframe)
        // overlay the result over the source video
        cv::addWeighted(frame, 1., dfram, overlay_slider, 0.0)

		// Display Image
        resize(dframe, rszd_frame, Size(640, 480), cv::INTER_LINEAR);
		imshow(window_name, rszd_frame);

        // TODO add the handle function later
        // Exit if ESC pressed
        int key_event = waitKey(1);
        if(key_event == 27) 
        {
            break;
        }
    }
    
    return 0;
}