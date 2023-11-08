@echo off
cls
setlocal ENABLEEXTENSIONS


rem Install Cmake
https://github.com/Kitware/CMake/releases/latest

if NOT exist /c/lib (
    mkdir /c/lib
    cd /c/lib
)


rem Check the platform and version to run the correct command line
; 32-bit system:
set 


CMAKE_GENERATOR_OPTIONS=-G"Visual Studio 14 2015 Win64"