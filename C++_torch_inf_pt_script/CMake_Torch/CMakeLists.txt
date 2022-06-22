# CMakeList.txt : CMake project for CMake_Torch, include source and define
# project specific logic here.
#
cmake_minimum_required (VERSION 3.8 FATAL_ERROR)


project(CMake_Torch)
find_package( OpenCV REQUIRED )


message(STATUS "OpenCV library status:")
message(STATUS "    config: ${OpenCV_DIR}")
message(STATUS "    version: ${OpenCV_VERSION}")
message(STATUS "    libraries: ${OpenCV_LIBS}")
message(STATUS "    include path: ${OpenCV_INCLUDE_DIRS}")
include_directories( ${OpenCV_INCLUDE_DIRS} )


find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

add_executable (CMake_Torch "CMake_Torch.cpp" "CMake_Torch.h")

target_link_libraries( CMake_Torch ${OpenCV_LIBS})
target_link_libraries(CMake_Torch "${TORCH_LIBRARIES}")
set_property(TARGET CMake_Torch PROPERTY CXX_STANDARD 14)



# The following code block is suggested to be used on Windows.
# According to https://github.com/pytorch/pytorch/issues/25457,
# the DLLs need to be copied to avoid memory errors.
if (MSVC)
  file(GLOB TORCH_DLLS "${TORCH_INSTALL_PREFIX}/lib/*.dll")
  add_custom_command(TARGET CMake_Torch
                     POST_BUILD
                     COMMAND ${CMAKE_COMMAND} -E copy_if_different
                     ${TORCH_DLLS}
                     $<TARGET_FILE_DIR:CMake_Torch>)
endif (MSVC)
if (MSVC)
  file(GLOB OpenCV_DLLS "${OpenCV_DIR}/bin/Release/*.dll")
  add_custom_command(TARGET CMake_Torch
                     POST_BUILD
                     COMMAND ${CMAKE_COMMAND} -E copy_if_different
                     ${OpenCV_DLLS}
                     $<TARGET_FILE_DIR:CMake_Torch>)

endif (MSVC)




# TODO: Add tests and install targets if needed.
