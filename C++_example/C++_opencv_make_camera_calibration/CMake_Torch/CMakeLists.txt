# CMakeList.txt : CMake project for CMake_Torch, include source and define
# project specific logic here.
#
cmake_minimum_required (VERSION 3.8 FATAL_ERROR)


project(CMake_ex)
find_package( OpenCV REQUIRED )


message(STATUS "OpenCV library status:")
message(STATUS "    config: ${OpenCV_DIR}")
message(STATUS "    version: ${OpenCV_VERSION}")
message(STATUS "    libraries: ${OpenCV_LIBS}")
message(STATUS "    include path: ${OpenCV_INCLUDE_DIRS}")
include_directories( ${OpenCV_INCLUDE_DIRS} )



set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")

add_executable (CMake_ex "camera_calibration.cpp" "camera_calibration.h")

target_link_libraries( CMake_ex ${OpenCV_LIBS})

set_property(TARGET CMake_ex PROPERTY CXX_STANDARD 14)


if (MSVC)
  file(GLOB OpenCV_DLLS "${OpenCV_DIR}/bin/Release/*.dll")
  add_custom_command(TARGET CMake_ex
                     POST_BUILD
                     COMMAND ${CMAKE_COMMAND} -E copy_if_different
                     ${OpenCV_DLLS}
                     $<TARGET_FILE_DIR:CMake_ex>)

endif (MSVC)




# TODO: Add tests and install targets if needed.
