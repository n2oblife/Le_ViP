# Live Video Processing (LeVIP)

## Introduction
This repository is the baseline of a C++ live video processing app. It uses [OpenCV](https://opencv.org/) to process images and works on both Windows and Linux.

## Installation
### Cloning the project
You can clone the projet directly from github by graphical interface or with CLI:
```
git clone https://github.com/n2oblife/live_video_processing.git
cd live_video_processing
```

### Setup your computer
To use this project you need t setup your computer. To do so you can use the scripts in the /scripts folder.
PS : you can take a coffee or two during the setup

#### Windows -
You can just click on the ```setup_project_win.bat``` file to install everything needed. You must follow the instruction and install everything by yourself until everything is done.

#### Linux (Ubuntu) -
You can just launch the ```setup_project_linux.sh``` file to install everything needed.

### Building the project
Once your computer is ready, use the shell/powershell to build the project as following.
```
mkdir build
cd build
cmake ..
make
```
If you want to rebuild your project after modifications juste enter ```make``` inside of the ```/build``` foler.
If you made some mistakes, just delete the folder and do the building again.
