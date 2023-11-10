# Version adapted for OpenCV
cmake_minimum_required(VERSION 3.9)


# Project name and a few useful settings. Other commands can pick up the results
project(
  LiveVideoProcessing
  VERSION 0.1
  DESCRIPTION "A project to display live video streaming while processing it"
  LANGUAGES CXX)


# Only do these if this is the main project, and not if it is included through add_subdirectory
if(CMAKE_PROJECT_NAME STREQUAL PROJECT_NAME)

  # Optionally set things like CMAKE_CXX_STANDARD, CMAKE_POSITION_INDEPENDENT_CODE here
  set(CMAKE_CXX_STANDARD 11)
  # set(CMAKE_VERBOSE_MAKEFILE true)

  # Let's ensure -std=c++xx instead of -std=g++xx
  set(CMAKE_CXX_EXTENSIONS OFF)

  # Let's nicely support folders in IDEs
  set_property(GLOBAL PROPERTY USE_FOLDERS ON)

  # Set the architecture specificities according to Intel NUC11TNKi5a
  set(CPU_BASELINE=SSE4_2)
  set(CPU_BASELINE=AVX2) # max available for OpenCV
  

  # Testing only available if this is the main app
  # Note this needs to be done in the main CMakeLists
  # since it calls enable_testing, which must be in the
  # main CMakeLists.
  include(CTest)

  # Docs only available if this is the main app
  find_package(Doxygen)
  if(Doxygen_FOUND)
    add_subdirectory(docs)
  else()
    message(STATUS "Doxygen not found, not building docs")
  endif()

  # Arguments for using OpenCV
  # set(OPENCV_DIR "C:/lib/build_opencv")
  set(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} ${GCC_OPENCV_DIR}")
  # set(OpenCV_PREFIX "C:/lib/build_opencv")
  # add only the used libraries
  find_package( OpenCV REQUIRED core highgui videoio )
  # PATHS ${OPENCV_DIR} )
  include_directories( ${OpenCV_INCLUDE_DIRS} ) # provide library headers
  MESSAGE(STATUS "OpenCV_LIBS: " ${OpenCV_LIBS} )  #display opencv libs found
  MESSAGE(STATUS "OpenCV path : " $(OpenCV_INSTALL_PATH))
  set(IMPORTED_IMPLIB_)

endif()

# FetchContent added in CMake 3.11, downloads during the configure step
# FetchContent_MakeAvailable was added in CMake 3.14; simpler usage
include(FetchContent)

# Accumulator library
# This is header only, so could be replaced with git submodules or FetchContent
find_package(Boost REQUIRED)
# Adds Boost::boost

# Formatting library
FetchContent_Declare(
  fmtlib
  GIT_REPOSITORY https://github.com/fmtlib/fmt.git
  GIT_TAG 5.3.0)
FetchContent_MakeAvailable(fmtlib)
# Adds fmt::fmt

# The compiled library code is here
add_subdirectory(src)

# The executable code is here
add_subdirectory(apps)

# Testing only available if this is the main app
# Emergency override MODERN_CMAKE_BUILD_TESTING provided as well
if((CMAKE_PROJECT_NAME STREQUAL PROJECT_NAME OR MODERN_CMAKE_BUILD_TESTING)
   AND BUILD_TESTING)
  add_subdirectory(tests)
endif()
