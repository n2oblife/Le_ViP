#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>
#include <string>

using namespace cv;
using namespace std;

int main( int argc, char** argv )
{
    std::string imageAdress = "/mnt/c/Users/ZaccarieKanit/Pictures/profil_pic.jpg";
    if( argc != 2)
    {
     std::cout <<" Usage: " << argv[0] << " ImageToLoadAndDisplay" << std::endl;
     return -1;
    }
    cv::Mat image = cv::imread(imageAdress, IMREAD_COLOR); // Read the file;
    // image = imread(argv[1], IMREAD_COLOR); // Read the file
    // image = imread(imageAdress, IMREAD_COLOR); // Read the file

    if( image.empty() ) // Check for invalid input
    {
        std::cout << "Could not open or find the image" << std::endl ;
        return -1;
    }
    cv::namedWindow( "Display window", WINDOW_AUTOSIZE ); // Create a window for display.
    cv::imshow( "Display window", image ); // Show our image inside it.
    cv::waitKey(0); // Wait for a keystroke in the window
    return 0;
}