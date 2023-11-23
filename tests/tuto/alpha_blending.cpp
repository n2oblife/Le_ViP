// #include <image_processing/alpha_blending.hpp>

#include <string>
#include <opencv2/videoio.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>

using namespace cv;
using namespace std;
 
void help(char** argv)
{
        printf("\n"
            "This program blends a foreground image into a bavkground image using an alpha filter with values between 0-1.\n"
            "Usage : \n" );
            // argv[0], "<path to foreground> <path to background> <path to alpha layer> \n");
}

// To initialize a matrix with solid color =>
// cv::Mat::setTo(cv::Scalar(redVal,greenVal,blueVal), InputArray mask=noArray());

void alphaBlend(Mat& foreground,Mat& background,Mat& alpha, Mat& outImage)
{
    // Find the total number of pixels in the images (assuming 2D matrices)
    int numberOfPixels = foreground.rows * foreground.cols * foreground.channels();

    // Get floating point pointers to the data matrices
    float* fptr = reinterpret_cast<float*>(foreground.data);
    float* bptr = reinterpret_cast<float*>(background.data);
    float* aptr = reinterpret_cast<float*>(alpha.data);
    float* outImagePtr = reinterpret_cast<float*>(outImage.data);

    // Loop over all pixels ONCE (can be parallelized)
    for (int i = 0; i < numberOfPixels; i++, outImagePtr++, fptr++, aptr++, bptr++)
    {
        // Perform alpha blending equation: result = (foreground * alpha) + (background * (1 - alpha))
        *outImagePtr = (*fptr) * (*aptr) + (*bptr) * (1 - *aptr);
    }
}

int main(int argc, char** argv)
{
    const string keys = 
        "{help h        | | Usage}"
        "{foreground f  | | The foreground image to use}"
        "{background b  | | The background image to use}"
        "{alpha a       | | The alpha image to use}"
        "{out o         | | The output image to save the blending in}";
    CommandLineParser parser(argc, argv, keys);

    // Display help if requested
    if (parser.has("help"))
    {
        help(argv);
        return 0;
    }
 
    // Read the images
    VideoCapture capFgnd, capBkckngd;
    capFgnd.open(samples::findFileOrKeep(parser.get<String>("foreground")));
    capBkckngd.open(samples::findFileOrKeep(parser.get<String>("background")));
    // capAlpha.open(samples::findFileOrKeep(parser.get<String>("alpha")));

    Mat tmp, foreground, background, alpha;
    // capAlpha >> tmp; 
    // resize(tmp, alpha, Size(640, 480), INTER_LINEAR);
    tmp = imread(parser.get<string>("alpha"));
    resize(tmp, alpha, Size(640, 480), INTER_LINEAR);
    // Normalize the alpha mask to keep intensity between 0 and 1
    alpha.convertTo(alpha, CV_32FC3, 1.0/255);


    for(;;)
    {

        capFgnd >> tmp; 
        resize(tmp, foreground, Size(640, 480), INTER_LINEAR);
        capBkckngd >> tmp; 
        resize(tmp, background, Size(640, 480), INTER_LINEAR);

        // Convert Mat to 3 channel of 32 bits float data type
        foreground.convertTo(foreground, CV_32FC3);
        background.convertTo(background, CV_32FC3);
    
    
        // Storage for output image
        Mat ouImage = Mat::zeros(foreground.size(), foreground.type());
    
        // Multiply the foreground with the alpha matte
        multiply(alpha, foreground, foreground); 
    
        // Multiply the background with ( 1 - alpha )
        multiply(Scalar::all(1.0)-alpha, background, background); 
    
        // Add the masked foreground and background.
        add(foreground, background, ouImage); 
    
        // Display image
        imshow("alpha blended image", ouImage/255);
        if(waitKey(1) == 27)
        {
            // Save the outImage
            if (! parser.get<string>("output").empty())
            {
                // ... save the image at the right place
            }
            break;
        }
    }
 
    return 0;
}