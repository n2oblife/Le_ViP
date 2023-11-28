# Testing library
FetchContent_Declare(
  catch
  GIT_REPOSITORY https://github.com/catchorg/Catch2.git
  GIT_TAG v2.13.6)
FetchContent_MakeAvailable(catch)
# Adds Catch2::Catch2

# Tests need to be added as executables first
add_executable(testlib testlib.cpp)

# I'm using C++17 in the test
target_compile_features(testlib PRIVATE cxx_std_17)

# Should be linked to the main library, as well as the Catch2 testing library
target_link_libraries(testlib PRIVATE modern_library Catch2::Catch2)

# add lib
# file(GLOB HEADER_LIST CONFIGURE_DEPENDS "${ModernCMakeExample_SOURCE_DIR}/include/modern/*.hpp")
# add_library( LVIP_LIB "../include/*")
# file(GLOB LVIP_LIB CMAKE_CONFIGURE_DEPENDS "{LiveVideoProcessing_SOURCE_DIR}/include/*.hpp")
# LiveVideoProcessing

# create prototype exec
add_executable(Proto prototype.cpp)
target_link_libraries(Proto PRIVATE ${OpenCV_LIBS})

# If you register a test, then ctest and make test will run it.
# You can also run examples and check the output, as well.
add_test(NAME testlibtest COMMAND testlib) # Command can be a target

add_subdirectory(tuto)