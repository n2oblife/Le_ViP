#pragma once

#include <string>
#include <opencv2/videoio.hpp>

/// @brief Determine which source to use for a VideoCapture class. 
/// Need to check : cap.isOpen()
/// @param input : a string input used as the source (if one character long it is considered an int)
/// @param cap   : a VideoCapture element to connect to the source
void determineInputVideoCap(
    const string& input, VideoCapture& cap
    );


void saveVideo()

/* a function to calculate the max frame of the video
because OpenCV can only save files up to 2Gb
file size = nbre frame * size frame
to put in the for loop when need to save*/
void videoMaxFrame()