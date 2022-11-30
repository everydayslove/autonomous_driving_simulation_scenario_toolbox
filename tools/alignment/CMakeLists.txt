cmake_minimum_required(VERSION 2.8)

project (ShakeImageWork)
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17 -pthread")

  
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../../third_party/opencv/include)
link_directories(${CMAKE_CURRENT_SOURCE_DIR}/../../third_party/opencv/lib/win64)

set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} /MT")
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} /MT")


link_libraries(
	opencv_world460.lib
	ade.lib
	IlmImf.lib
	ippicvmt.lib
	ippiw.lib
	ittnotify.lib
	libjpeg-turbo.lib
	libopenjp2.lib
	libpng.lib
	libprotobuf.lib
	libwebp.lib
	libtiff.lib
	quirc.lib
	zlib.lib
	)

set(SRC_LIST
	./dll/src/work.cpp
	./dll/src/display.cpp
	./dll/src/match.cpp
	./dll/src/sift.cpp
	./dll/src/ORBextractor.cc)

add_library(ShakeImage SHARED ${SRC_LIST})
add_executable(ShakeImageTest ./test/main.cpp)

