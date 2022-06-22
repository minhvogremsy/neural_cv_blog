# CMakeList.txt : CMake project for CMake_Torch, include source and define
# project specific logic here.
#
cmake_minimum_required(VERSION 3.10.0)

project(helloworld)

set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTORCC ON)
set(CMAKE_AUTOUIC ON)

set(CMAKE_INCLUDE_CURRENT_DIR ON)

find_package(Qt5 CONFIG REQUIRED COMPONENTS Widgets Network)


SET("OpenCV_DIR" "C:\\opencv\\build_455")
find_package( OpenCV REQUIRED )
message(STATUS "OpenCV library status:")
message(STATUS "    config: ${OpenCV_DIR}")
message(STATUS "    version: ${OpenCV_VERSION}")
message(STATUS "    libraries: ${OpenCV_LIBS}")
message(STATUS "    include path: ${OpenCV_INCLUDE_DIRS}")
include_directories( ${OpenCV_INCLUDE_DIRS})



message(STATUS "QT library status:")
message(STATUS "    config: ${Qt5_DIR}")
add_executable(helloworld
   mainwindow.ui
   mainwindow.cpp
   QT.cpp
   object.cpp
   resources.qrc networkthread.cpp)

target_link_libraries(helloworld ${OpenCV_LIBS}
   Qt5::Widgets Qt5::Network)


if (MSVC)
  file(GLOB OpenCV_DLLS "${OpenCV_DIR}/bin/Release/*.dll")
  add_custom_command(TARGET helloworld
                     POST_BUILD
                     COMMAND ${CMAKE_COMMAND} -E copy_if_different
                     ${OpenCV_DLLS}
                     $<TARGET_FILE_DIR:helloworld>)

endif (MSVC)



# TODO: Add tests and install targets if needed.
