#include <image_processing/frame/median_estimation.hpp>
#include <image_processing/utility/device.hpp>

#include <opencv2/core.hpp>
#include <iostream>
#include <random>

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
    const CAP_INPUT& input,
    int& nFrames = 25
)
{
	// init capture
    auto cap = initVideoCap(input);

	// Set the parameters
	nFrames = std::min(nFrames, cap.get(cv::CAP_PROP_FRAME_COUNT));
	std::vector<cv::Mat> frames;
	cv::Mat frame;

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

	// Calculate the median along the time axis
	return compute_median(frames);
}

cv::Mat computeMedianFrame(
	const cv::VideoCapture cap,
    int& nFrames = 25
)
{
	// Set the parameters
    nFrames = std::min(nFrames, cap.get(cv::CAP_PROP_FRAME_COUNT));
	std::vector<cv::Mat> frames;
	cv::Mat frame;

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

	// Calculate the median along the time axis
	return compute_median(frames);
}

cv::Mat computeMedianFrame(
    const CAP_INPUT& input,
	cv::Mat& frame,
	const bool& working=true,
    int& nFrames = 25
)
{
	// init capture
    auto cap = initVideoCap(input);
	
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
