#include <user_interface/interface/trackbar.hpp>

#include <opencv2/highgui.hpp>

// TODO later might be tailored for use

// Any type of value used as a slider (int, float, double etc.)
template <typename SL_TYPE>;

class trackbar
{
private:
    SL_TYPE slider_max=100, slider_min=0, slider;
    char slider_name[64], window_name[64];
public:
    trackbar(/* args */);
    ~trackbar();
};

trackbar::trackbar(
    char slider_name, 
    char window_name, 
    SL_TYPE slider_max=100,
    SL_TYPE slider_min=0,
    SL_TYPE slider 
    )
{
}

static void trackbar::on_trackbar(char )
{
    imsh
}

trackbar::~trackbar()
{
}
