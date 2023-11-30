#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include <iostream>
using namespace cv;
using namespace std;
Mat src_gray;
int thresh = 100;
RNG rng(12345);
void thresh_callback(int, void* );
int main( int argc, char** argv )
{
    CommandLineParser parser( argc, argv, "{@input | HappyFish.jpg | input image}" );
    Mat src = imread( samples::findFile( parser.get<String>( "@input" ) ) );
    Mat mask = imread(samples::findFile("../../../../datasets/transseptal_puncture_mask.png"));
    if( src.empty() )
    {
      cout << "Could not open or find the image!\n" << endl;
      cout << "Usage: " << argv[0] << " <Input image>" << endl;
      return -1;
    }
    cout << " --- BEFORE ---" << endl;
    cout << " MASK INFO : " << mask.type() << " & " << mask.size() << endl;
    cout << " SRC INFO : " << src.type() << " & " << src.size() << endl;
    // cvtColor( mask, mask, COLOR_BGR2GRAY );
    resize(mask, mask, src.size());
    mask.convertTo(mask, src.type(), 1.0/255);
    cout << " --- AFTER --- " << endl;
    cout << " MASK INFO : " << mask.type() << " & " << mask.size() << endl;
    cout << " SRC INFO : " << src.type() << " & " << src.size() << endl;
    multiply(src, mask, src);
    cvtColor( src, src_gray, COLOR_BGR2GRAY );
    blur( src_gray, src_gray, Size(1,1) );
    const char* source_window = "Source";
    namedWindow( source_window );
    resize(src, src, cv::Size(640, 480), INTER_LINEAR);
    imshow( source_window, src );
    const int max_thresh = 255;
    createTrackbar( "Canny thresh:", source_window, &thresh, max_thresh, thresh_callback );
    thresh_callback( 0, 0 );
    waitKey();
    return 0;
}
void thresh_callback(int, void* )
{
    Mat canny_output;
    Canny( src_gray, canny_output, thresh, thresh*2 );
    vector<vector<Point> > contours;
    vector<Vec4i> hierarchy;
    findContours( canny_output, contours, hierarchy, RETR_TREE, CHAIN_APPROX_SIMPLE );
    Mat drawing = Mat::zeros( canny_output.size(), CV_8UC3 );
    for( size_t i = 0; i< contours.size(); i++ )
    {
        Scalar color = Scalar( rng.uniform(0, 256), rng.uniform(0,256), rng.uniform(0,256) );
        drawContours( drawing, contours, (int)i, color, 2, LINE_8, hierarchy, 0 );
    }
    resize(drawing, drawing, Size(640, 480), INTER_LINEAR);
    imshow( "Contours", drawing );
}