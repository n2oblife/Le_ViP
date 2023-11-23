#pragma once 
#ifndef KEYBOARD_HANDLING_HPP
#define KEYBOARD_HANDLING_HPP

#include <opencv2/core.hpp>

/// @brief Handles the events when a key is pushed while computing.
/// To be used in loop of frame computation.
/// @param key The ASCII value of a char
void keyboardEvent(int& key);

/// @brief Handles the events when a key is pushed while computing.
/// To be used in loop of frame computation.
/// @param event The char key pressed 
void keyboardEvent(const char& event);

#endif // KEYBOARD_HANDLING_HPP