#pragma once 

#include <string>
#include <opencv2/core/mat.hpp>

static void help(char** argv);

/// @brief This function refines a segmentation mask using morphological operations and draws the largest contour on a destination Mat. 
/// It is used improve the quality of segmentation results. 
/// @param img  : input matrix with the image
/// @param mask : input mask for segmenting the image
/// @param dst  : output contours of the segmentation mask
static void refineSegments(const Mat& img, Mat& mask, Mat& dst);

