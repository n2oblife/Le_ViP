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
target_link_libraries(DNNObjDetect PRIVATE ${OpenCV_LIBS})

add_executable(DNNSegmentation dnn_segmentation.cpp)
target_link_libraries(DNNSegmentation PRIVATE ${OpenCV_LIBS})

add_executable(VideoWrite video_write.cpp)
target_link_libraries(VideoWrite PRIVATE ${OpenCV_LIBS})

add_executable(VideoWriteBasic video_write_basic.cpp)
target_link_libraries(VideoWriteBasic PRIVATE ${OpenCV_LIBS})

add_executable(ForegroundMasking foreground_masking.cpp)
target_link_libraries(ForegroundMasking PRIVATE ${OpenCV_LIBS})

add_executable(Tracker tracker.cpp)
target_link_libraries(Tracker PRIVATE ${OpenCV_LIBS})

add_executable(DeepSort deepsort.cpp)
target_link_libraries(DeepSort PRIVATE ${OpenCV_LIBS})

add_executable(MedianBg median_estimation.cpp)
target_link_libraries(MedianBg PRIVATE ${OpenCV_LIBS})

add_executable(Overlay alpha_blending.cpp)
target_link_libraries(Overlay PRIVATE ${OpenCV_LIBS})

add_executable(WaterShed watershed.cpp)
target_link_libraries(WaterShed PRIVATE ${OpenCV_LIBS})

add_executable(Contour find_contour.cpp)
target_link_libraries(Contour PRIVATE ${OpenCV_LIBS})

add_executable(Grab grabcut.cpp)
target_link_libraries(Grab PRIVATE ${OpenCV_LIBS})

add_executable(Canny canny_edge.cpp)
target_link_libraries(Canny PRIVATE ${OpenCV_LIBS})

add_executable(Lumen lumen_correction.cpp)
target_link_libraries(Lumen PRIVATE ${OpenCV_LIBS})

add_executable(Hough houg_lines.cpp)
target_link_libraries(Hough PRIVATE ${OpenCV_LIBS})

add_executable(OptFlo optical_flow.cpp)
target_link_libraries(OptFlo PRIVATE ${OpenCV_LIBS})
