add_executable( DisplayImage DisplayImage.cpp )
target_link_libraries( DisplayImage PRIVATE ${OpenCV_LIBS} Catch2::Catch2)
#add_test(NAME tuto_test COMMAND /mnt/c/Users/ZaccarieKanit/SIMULANDS/git_projects/prototype/live_video_processing/build/tests/tuto/DisplayImage /mnt/c/Users/ZaccarieKanit/Pictures/profil_pic.jpg)

add_executable( DisplayCam video_proc_U.cpp )
target_link_libraries( DisplayCam PRIVATE ${OpenCV_LIBS} Catch2::Catch2)

add_executable( FaceDetect face_detect.cpp)
target_link_libraries( FaceDetect PRIVATE ${OpenCV_LIBS})

add_executable(SegmentObj segment_object.cpp)
target_link_libraries( SegmentObj PRIVATE ${OpenCV_LIBS})

add_executable( DNNObjDetect dnn_object_detection.cpp)
target_link_libraries(DNNObjDetect PRIVATE ${OpenCV_LIBS} common)

add_executable(DNNSegmentation dnn_segmentation.cpp)
target_link_libraries(DNNSegmentation PRIVATE ${OpenCV_LIBS} common)

add_executable(VideoWrite video_write.cpp)
target_link_libraries(VideoWrite PRIVATE ${OpenCV_LIBS})

add_executable(VideoWriteBasic video_write_basic.cpp)
target_link_libraries(VideoWriteBasic PRIVATE ${OpenCV_LIBS})