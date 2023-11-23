#pragma once
#ifndef DEVICE_HPP
#define DEVICE_HPP

#include <string>
#include <opencv2/videoi.hpp>
using namespace std;
using namespace cv;

void determineInputVideoCap(const string& input, VideoCapture& cap);
void videoMaxFrame(const double& fps, const size& frameSize, int& maxFrame);
int videoMaxFrame(const double& fps, const size& frameSize, const int& frameRate);

#endif // DEVICE_HPP