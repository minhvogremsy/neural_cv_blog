# CMakeList.txt : CMake project for CMake_Torch, include source and define
# project specific logic here.
#
cmake_minimum_required (VERSION 3.10.0 FATAL_ERROR)
set(CMAKE_CXX_FLAGS "-latomic")

set(CMAKE_BUILD_TYPE Release)
SET(CMAKE_CXX_FLAGS_RELEASE "-O2")
##########################
set(CMAKE_AUTOMOC ON) #Qt targets.
set(CMAKE_AUTORCC ON) #Qt targets.
set(CMAKE_AUTOUIC ON) #Qt targets.
set(CMAKE_INCLUDE_CURRENT_DIR ON)
#############################
project(CMake_ex)

#set (CMAKE_PREFIX_PATH "C:\\Qt\\Qt5.12.12\\5.12.12\\msvc2017_64\\")
#SET("Qt5_DIR" "C:\\Qt\\Qt5.12.12\\5.12.12\\msvc2017_64")
find_package(Qt5 CONFIG REQUIRED COMPONENTS Widgets)


SET("OpenCV_DIR" "C:\\opencv\\build_455")
find_package( OpenCV REQUIRED )
message(STATUS "OpenCV library status:")
message(STATUS "    config: ${OpenCV_DIR}")
message(STATUS "    version: ${OpenCV_VERSION}")
message(STATUS "    libraries: ${OpenCV_LIBS}")
message(STATUS "    include path: ${OpenCV_INCLUDE_DIRS}")
include_directories( ${OpenCV_INCLUDE_DIRS} )


set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")

add_executable (CMake_ex "Thread.cpp" "Thread.h")

add_executable (CMake_ex_th "Mutex.cpp")
add_executable (GST_test "GST_test.cpp")
add_executable (QT "QT.cpp")
add_executable (panaram "opencv_panaram.cpp")
add_executable (get_camera "opencv_get.cpp")

target_link_libraries( GST_test ${OpenCV_LIBS})
target_link_libraries( panaram ${OpenCV_LIBS})


set_property(TARGET CMake_ex PROPERTY CXX_STANDARD 14)
set_property(TARGET panaram PROPERTY CXX_STANDARD 14)
set_property(TARGET CMake_ex_th PROPERTY CXX_STANDARD 14)
set_property(TARGET GST_test PROPERTY CXX_STANDARD 14)
set_property(TARGET QT PROPERTY CXX_STANDARD 14)

if (MSVC)
  file(GLOB OpenCV_DLLS "${OpenCV_DIR}/bin/Release/*.dll")
  add_custom_command(TARGET GST_test
                     POST_BUILD
                     COMMAND ${CMAKE_COMMAND} -E copy_if_different
                     ${OpenCV_DLLS}
                     $<TARGET_FILE_DIR:GST_test>)

endif (MSVC)




# TODO: Add tests and install targets if needed.
