@if (@a==@b) @end /*

:: setup_project.bat
:: setting up the project and cmake library made for windows

cls
@echo off
setlocal ENABLEEXTENSIONS

ECHO Checking Cmake
WHERE cmake
IF %ERRORLEVEL% NEQ 0 (
    ECHO CMake wasn't found, go to https://github.com/Kitware/CMake/releases/latest and install CMake
    ECHO Download the .msi file, launch the executable and follow the instructions
    GOTO :EOF
)

ECHO Checking git-bash
WHERE git
IF %ERRORLEVEL% NEQ 0 (
    ECHO git wasn't found, go to https://gitforwindows.org/ and install git-bash
    ECHO Download the .exe file, launch the executable and follow the instructions
    GOTO :EOF
)

rem Check if /c/lib/ directory exists to copy the OpenCV files in it
IF NOT EXIST C:\lib\ (
    mkdir C:\lib\
)

rem Check the cmake option according to the VSCode version and architecture 
rem variable to do "or" statement
SET temp_bool=F
SET answer=y
SET /P answer=Have you adapted the CMAKE_GENERATOR_OPTIONS in the ./installOCV.sh file ? [Y/n] 
IF /I %answer%==y SET temp_bool=T
IF /I %answer%==yes SET temp_bool=T
IF %temp_bool%==F (
    ECHO Go change line 6 in the file to adapt the installation to your platform
    ECHO You can go to the help panel in Visual Studio to see the version
    GOTO :EOF
)

rem moving the installation file to the lib folder and launch file
SET bat_path=%~dp0
IF NOT EXIST C:\lib\installOCV.sh (
    xcopy  %bat_path:~0%installOCV.sh C:\lib\
)

cd C:\lib\
ECHO Running the OpenCV installation executable ...
bash installOCV.sh
IF %ERRORLEVEL% NEQ 0 (
    ECHO Install bash to launch .sh file : https://git-scm.com/download/win
    ECHO Download the .exe file, launch the executable and follow the instructions
    GOTO :EOF
)

