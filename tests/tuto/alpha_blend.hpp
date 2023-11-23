#pragma once

#include <string>
#include <opencv2/videoi.hpp>

void alphaBlend(Mat& foreground, Mat& background, Mat& alpha, Mat& outImage);

