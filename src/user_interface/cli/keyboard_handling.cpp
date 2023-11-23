#include <user_interface/cli/keyboard_handling.hpp>

#include <opencv2/highgui.hpp>

// to be used in loop of frame computation
void keyboardEvent(int& key)
{
    // Esc is pressed
    if (key == 27)
    {
        break;
    }
}