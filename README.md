# Live Video Processing (Le_ViP)

## Introduction
This repository is the baseline of a C++ live video processing app. It uses [OpenCV](https://opencv.org/) to process images and works on both Windows and Linux.

It also uses the [json modern lib](https://github.com/nlohmann/json) to parse files.

## Installation
### Cloning the project
You can clone the projet directly from github by graphical interface or with CLI:
```
git clone https://github.com/n2oblife/live_video_processing.git
cd live_video_processing
```

### Setup your computer
To use this project you need t setup your computer.
PS : you can take a coffee or two during the setup

#### Windows -

You have to install some things :
- [C++ compiler](https://learn.microsoft.com/en-us/cpp/windows/latest-supported-vc-redist?view=msvc-170)
- [CMake](https://cmake.org/download/)
- [Boost](https://www.boost.org/)
- [GStreamer](https://gstreamer.freedesktop.org/documentation/installing/on-windows.html?gi-language=c)

When everything is set up, install [OpenCV](https://opencv.org/get-started/) (the Windows' release version) and put it in ```C:\lib``` and then launch ```C:\lib\opencv\build\setup_vars_opencv4.cmd``` to setup the environnement variables automatically.


#### Linux (Ubuntu) -
You can just launch the ```setup_project_linux.sh``` file from terminal to install everything needed.

### Building the project
Once your computer is ready, use the shell/powershell to build the project as following.
```
mkdir build
cd build
cmake ..
make --build . --config <release or debug> -j <nbr of threads>
```
To have the number of thread on Linux : ```$(nproc)```

To have the number of thread on Windows : [Intel_FAQ](https://www.intel.com/content/www/us/en/support/articles/000029254/processors.html)

If you want to rebuild your project after modifications juste enter the last line inside of the ```/build``` folder.
If you changed the architecture of the project or one of the CMakeList.txt or you made some mistakes, just delete the ```/build``` folder and do the building again.
