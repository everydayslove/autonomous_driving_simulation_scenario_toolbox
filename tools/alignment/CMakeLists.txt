cmake_minimum_required(VERSION 2.8)

project (ShakeImageWork)
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17 -pthread")
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/opencv-4.x/include)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/opencv-4.x/include/opencv2)

link_directories(${CMAKE_CURRENT_SOURCE_DIR}/opencv-4.x/lib)

link_libraries(
	#libboost_system.so
	opencv_calib3d460d.lib 
	opencv_core460d.lib
	opencv_dnn460d.lib
	opencv_features2d460d.lib
	opencv_flann460d.lib
	opencv_gapi460d.lib
	opencv_highgui460d.lib
	opencv_imgcodecs460d.lib
	opencv_imgproc460d.lib
	opencv_ml460d.lib
	opencv_objdetect460d.lib
	opencv_photo460d.lib
	opencv_stitching460d.lib
	opencv_video460d.lib
	opencv_videoio460d.lib
	)

set(SRC_LIST
	./dll/src/work.cpp
	./dll/src/main.cpp
	./dll/src/display.cpp
	./dll/src/match.cpp
	./dll/src/sift.cpp
	./dll/src/ORBextractor.cc)

add_library(ShakeImage SHARED ${SRC_LIST})
add_executable(ShakeImageTest ./test/main.cpp)


