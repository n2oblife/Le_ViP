#include <image_processing/frame/image_correction.hpp>

namespace vip
{
    template <class M = cv::Mat> 
    void lumenCorrection(
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


    void lumenCorrectionBGR(
        cv::InputOutputArray frame,
        cv::InputArray& src_frame,
        cv::OutputArray& out_frame
    )
    {
        // Init variables
        cv::InputOutputArray tmp_frame(src_frame.getMat());

        // READ RGB color image and convert it to Lab
        cv::cvtColor(src_frame, tmp_frame, cv::COLOR_BGR2Lab);

        // Extract the L channel
        std::vector<cv::Mat> lab_planes(3);
        cv::split(tmp_frame, lab_planes);  // now we have the L image in lab_planes[0]

        // apply the CLAHE algorithm to the L channel
        cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE();
        clahe->setClipLimit(4);
        clahe->apply(lab_planes[0], tmp_frame);

        // Merge the the color planes back into an Lab image
        tmp_frame.copyTo(lab_planes[0]);
        cv::merge(lab_planes, tmp_frame);
        cv::cvtColor(tmp_frame, out_frame, cv::COLOR_Lab2BGR);
    }


    void lumenCorrectionLab(
        cv::Mat& src_frame,
        cv::Mat& out_frame
    )
    {
        // Init variables
        cv::Mat tmp_frame;

        // Extract the L channel
        std::vector<cv::Mat> lab_planes(3);
        cv::split(tmp_frame, lab_planes);  // now we have the L image in lab_planes[0]

        // apply the CLAHE algorithm to the L channel
        cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE();
        clahe->setClipLimit(4);
        clahe->apply(lab_planes[0], tmp_frame);

        // Merge the the color planes back into an Lab image
        tmp_frame.copyTo(lab_planes[0]);
        cv::merge(lab_planes, tmp_frame);
        cv::cvtColor(tmp_frame, out_frame, cv::COLOR_Lab2BGR);
    }


    void thresholdedGaussianBlur(
        cv::Mat& src_frame, 
        cv::Mat& out_frame,
        const double& threshold=45.,
        const int& niters=2
    )
    {
        cv::Mat tmp(src_frame);
        cv::GaussianBlur(tmp, tmp, cv::Size(5,5),0 , 2);
        cv::threshold(tmp, tmp, threshold, 255, cv::THRESH_BINARY);
        cv::erode(tmp,tmp, cv::Mat(), cv::Point(-1,-1), niters);
        cv::dilate(tmp, out_frame, cv::Mat(), cv::Point(-1, -1), niters);
    }

    void thresholdedGaussianBlurToGray(
        cv::Mat& src_frame, 
        cv::Mat& out_frame,
        const double& threshold = 45.
    )
    {
        cv::Mat tmp(src_frame);
        cv::GaussianBlur(tmp, tmp, cv::Size(5,5),0 , 2);
        cv::threshold(tmp, tmp, threshold, 255, cv::THRESH_BINARY);
        cv::erode(tmp,tmp, cv::Mat(), cv::Point(-1,-1), 2);
        cv::dilate(tmp, out_frame, cv::Mat(), cv::Point(-1, -1), 2);
        cv::cvtColor(tmp, tmp, cv::COLOR_BGR2GRAY);
    }

} // namespace vip

