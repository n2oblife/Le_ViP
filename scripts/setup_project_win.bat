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


@REM rem Check the cmake option according to the VSCode version and architecture 
@REM rem variable to do "or" statement
@REM SET temp_bool=F
@REM SET answer=y
@REM SET /P answer=Have you adapted the CMAKE_GENERATOR_OPTIONS in the ./installOCV.sh file ? [Y/n] 
@REM IF /I %answer%==y SET temp_bool=T
@REM IF /I %answer%==yes SET temp_bool=T
@REM IF %temp_bool%==F (
@REM     ECHO Go change line 6 in the file to adapt the installation to your platform
@REM     ECHO You can go to the help panel in Visual Studio to see the version
@REM     GOTO :EOF
@REM )

@REM rem moving the installation file to the lib folder and launch file
@REM SET bat_path=%~dp0
@REM IF NOT EXIST C:\lib\installOCV.sh (
@REM     xcopy  %bat_path:~0%installOCV.sh C:\lib\
@REM )

rem moving the installation file to the lib folder and launch file
SET bat_path=%~dp0
IF NOT EXIST C:\lib\setup_project_linux.sh (
    xcopy  %bat_path:~0%setup_project_linux.sh C:\lib\
)

cd C:\lib\
ECHO Running the OpenCV installation executable ...
bash setup_project_linux.sh
IF %ERRORLEVEL% NEQ 0 (
    ECHO Install bash to launch .sh file : https://git-scm.com/download/win
    ECHO Download the .exe file, launch the executable and follow the instructions
    GOTO :EOF
)

