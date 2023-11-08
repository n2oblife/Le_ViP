add_executable( DisplayImage DisplayImage.cpp )
target_link_libraries( DisplayImage PRIVATE ${OpenCV_LIBS} Catch2::Catch2)
#add_test(NAME tuto_test COMMAND /mnt/c/Users/ZaccarieKanit/SIMULANDS/git_projects/prototype/live_video_processing/build/tests/tuto/DisplayImage /mnt/c/Users/ZaccarieKanit/Pictures/profil_pic.jpg)

add_executable( DisplayCam live_video_processing_yt.cpp )
target_link_libraries( DisplayCam PRIVATE ${OpenCV_LIBS} Catch2::Catch2)