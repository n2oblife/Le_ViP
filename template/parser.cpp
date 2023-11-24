#include <opencv2/core/utility.hpp>

/// To define the CommandLine Parser

const char* params
    = "{ help h         |           | Print usage }"
      "{ input1 i1      |           | Help message for inut1 }" // default value will be ""
      "{ input2 i2      | default2  | Help message for inut2 }" // default value will be "default2"
      "{ input3 i3      | <none>    | Help message for inut3 }" // no default value, returned key must not be empty
      "{ @input4        |           | Help message for inut4 }" // value can be given without key, first non key element 
      ;


int main(int argc, char* argv[])
{
    CommandLineParser parser(argc, argv, params);
    parser.about( "This program shows how to use CommandLine Parser by OpenCV.\n");
    if (parser.has("help"))
    {
        //print about message and help information
        parser.printMessage();
    }
}


/// To use in CLI

//>> ./script.cpp -i1=truc -i2="autre truc" "dernier truc"