cmake_minimum_required( VERSION 3.2 )
project( CVProject3 )

SET(CMAKE_C_COMPILER gcc)
SET(CMAKE_CXX_COMPILER g++)

set( CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${PROJECT_SOURCE_DIR}/cmake-modules )
set( CMAKE_CXX_STANDARD 14 )

find_package( Threads REQUIRED )
find_package( OpenCV REQUIRED )
find_package( OpenMP REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS} )

set( UTIL_SOURCES utils/FrameReader.cpp utils/ImageShower.h )

add_executable( CVProject3 main.cpp ${UTIL_SOURCES} )
target_link_libraries( CVProject3 ${OpenCV_LIBS} Threads::Threads ${OpenMP_CXX_LIBRARIES} ${OpenMP_C_LIBRARIES} )
