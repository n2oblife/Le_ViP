#include <image_processing/frame/image_correction.hpp>

// TODO remove the temp frames, swap vectors with arrays


template <class M> 
void vip::lumenCorrection(
    M &src_frame,
    M &out_frame, 
    const bool& isColor = true
)
{
    // Init variables
    M tmp_frame;
    int f_transf, l_transf;
    
    // TODO check other cases
    if (isColor)
    {
        f_transf = cv::COLOR_BGR2Lab, l_transf = cv::COLOR_Lab2BGR;
        // READ RGB color image and convert it to Lab
        cv::cvtColor(src_frame, tmp_frame, f_transf);
    }

    // Extract the L channel
    std::array<3, M> lab_planes;
    cv::split(tmp_frame, lab_planes);  // now we have the L image in lab_planes[0]

    // apply the CLAHE algorithm to the L channel
    cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE();
    clahe->setClipLimit(4);
    M dst;
    clahe->apply(lab_planes[0], tmp_frame);

    // Merge the the color planes back into an Lab image
    tmp_frame.copyTo(lab_planes[0]);
    cv::merge(lab_planes, tmp_frame);
    cv::cvtColor(tmp_frame, out_frame, l_transf);
}

template <class M>
void vip::lumenCorrectionBGR(
    M &src_frame,
    M &out_frame
)
{
    // READ RGB color image and convert it to Lab
    cv::cvtColor(src_frame, out_frame, cv::COLOR_BGR2Lab);

    // Extract the L channel
    std::array<M, 3> lab_planes;
    cv::split(out_frame, lab_planes);  // now we have the L image in lab_planes[0]

    // apply the CLAHE algorithm to the L channel
    cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE();
    clahe->setClipLimit(4);
    clahe->apply(lab_planes[0], out_frame);

    // Merge the the color planes back into an Lab image
    out_frame.copyTo(lab_planes[0]);
    cv::merge(lab_planes, out_frame);
    cv::cvtColor(out_frame, out_frame, cv::COLOR_Lab2BGR);
}

template <class M>
void vip::lumenCorrectionLab(
    M &src_frame,
    M &out_frame
)
{
    // Initilize the variables 
    src_frame.copyTo(out_frame);

    // Extract the L channel
    std::array<M, 3> lab_planes;
    cv::split(src_frame, lab_planes);  // now we have the L image in lab_planes[0]

    // apply the CLAHE algorithm to the L channel
    cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE();
    clahe->setClipLimit(4);
    clahe->apply(lab_planes[0], src_frame);

    // Merge the the color planes back into an Lab image
    src_frame.copyTo(lab_planes[0]);
    cv::merge(lab_planes, src_frame);
    cv::cvtColor(src_frame, out_frame, cv::COLOR_Lab2BGR);
}

template <class M>
void vip::thresholdedGaussianBlur(
    M &src_frame, 
    M &out_frame,
    const double& threshold=45.,
    const int& niters=2
)
{
    // Initialize 
    src_frame.copyTo(out_frame);
    cv::GaussianBlur(out_frame, out_frame, cv::Size(5,5),0 , 2);
    cv::threshold(out_frame, out_frame, threshold, 255, cv::THRESH_BINARY);
    cv::erode(out_frame,out_frame, cv::Mat(), cv::Point(-1,-1), niters);
    cv::dilate(out_frame, out_frame, cv::Mat(), cv::Point(-1, -1), niters);
}

template <class M>
void vip::thresholdedGaussianBlurToGray(
    M &src_frame, 
    M &out_frame,
    const double& threshold = 45.
)
{
    // Initialize the values
    src_frame.copyTo(out_frame);
    cv::GaussianBlur(out_frame, out_frame, cv::Size(5,5),0 , 2);
    cv::threshold(out_frame, out_frame, threshold, 255, cv::THRESH_BINARY);
    cv::erode(out_frame,out_frame, cv::Mat(), cv::Point(-1,-1), 2);
    cv::dilate(out_frame, out_frame, cv::Mat(), cv::Point(-1, -1), 2);
    cv::cvtColor(out_frame, out_frame, cv::COLOR_BGR2GRAY);
}

