#!/bin/bash -e

# Install minimal prerequisites (Ubuntu 18.04 as reference)
sudo apt update && sudo apt upgrade -y
sudo apt install -y build-essential cmake g++ wget unzip gcc

# Install Boost
sudo apt -y install libboost-all-dev

# Install FFMPEG
sudo apt -y install ffmpeg

# Install GStreamer
sudo apt -y install libgstreamer1.0-dev libgstreamer-plugins-base1.0-dev libgstreamer-plugins-bad1.0-dev gstreamer1.0-plugins-base gstreamer1.0-plugins-good gstreamer1.0-plugins-bad gstreamer1.0-plugins-ugly gstreamer1.0-libav gstreamer1.0-tools gstreamer1.0-x gstreamer1.0-alsa gstreamer1.0-gl gstreamer1.0-gtk3 gstreamer1.0-qt5 gstreamer1.0-pulseaudio

# Install Eigen3
sudo apt -y install libeigen3-dev

# Install OpenCV
cd
mkdir lib && cd lib

wget -O opencv.zip https://github.com/opencv/opencv/archive/4.x.zip
wget -O opencv_contrib.zip https://github.com/opencv/opencv_contrib/archive/4.x.zip
unzip opencv.zip
unzip opencv_contrib.zip
# Create build directory and switch into it
mkdir -p opencv && cd opencv
# Configure
cmake -DOPENCV_EXTRA_MODULES_PATH=../opencv_contrib-4.x/modules ../opencv-4.x
# Build
cmake --build . -j $(nproc)
cd ..
rm -r -f opencv.zip opencv-4.x opencv_contrib.zip opencv_contrib-4.x