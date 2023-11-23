#include <user_interface/cli/keyboard_handling.hpp>

#include <opencv2/highgui.hpp>


// ASCII conversion
// 1-9 : 49-57
// a-z : 97-122
// + : 43
// - : 45
// / : 47
// * : 42
// . : 46

void keyboardEvent(const int& key)
{
    switch (key)
    {
    // Esc
    case 27:
        return break;
    
    default:
        return;
    }
}

void keyboardEvent(const char& event)
{
    const int key = int(event);
    switch (key)
    {
    // Esc
    case 27:
        return break;
    
    default:
        return;
    }
}
