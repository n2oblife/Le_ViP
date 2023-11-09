@if (@a==@b) @end /*

:: setup_project.bat
:: setting up the project and cmake library made for windows

cls
@echo off
setlocal ENABLEEXTENSIONS

rem Check and/or install Cmake
WHERE cmake
IF %ERRORLEVEL% NEQ 0 (
    ECHO CMake wasn't found, go to https://github.com/Kitware/CMake/releases/latest and install CMake
    ECHO Download the .msi file and launch the executable
    GOTO : EOF
)

rem Check and/or install git-bash
WHERE git
IF %ERRORLEVEL% NEQ 0 (
    ECHO git wasn't found, go to https://gitforwindows.org/ and install git-bash
    GOTO : EOF
)

rem Check if /c/lib/ directory exists to copy the OpenCV files in it
IF NOT EXIST /c/lib (
    mkdir /c/lib
)

SET bat_path=%~dp0
IF NOT EXIST /c/lib/installOCV.sh (
    mv %bat_path:~0%installOCV.sh /c/lib
)

cd /c/lib

ECHO DONE 
GOTO : EOF



CMAKE_GENERATOR_OPTIONS=-G"Visual Studio 14 2015 Win64"
