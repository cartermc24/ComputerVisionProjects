cmake_minimum_required( VERSION 3.2 )
project( CVProject1 )

set( CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${PROJECT_SOURCE_DIR}/cmake-modules )
set( CMAKE_CXX_STANDARD 14 )

find_package( Threads REQUIRED )
find_package( OpenCV REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS} )

set( UTIL_SOURCES utils/FrameReader.cpp utils/ImageShower.h )

add_executable( CVProject1 main.cpp ${UTIL_SOURCES} )
target_link_libraries( CVProject1 ${OpenCV_LIBS} Threads::Threads )