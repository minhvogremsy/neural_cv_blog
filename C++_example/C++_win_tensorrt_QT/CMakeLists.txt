# cmake needs this line
cmake_minimum_required(VERSION 3.10)
# cmake_minimum_required(VERSION 2.8)

# Enable C++11
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)

# Define project name
project(trt_efficientdet LANGUAGES CXX CUDA)

SET (CMAKE_VERBOSE_MAKEFILE 1)

# Compile options
set(CMAKE_C_FLAGS "-Wall -pthread ")
set(CMAKE_C_FLAGS_DEBUG "-g -O0")
set(CMAKE_C_FLAGS_RELEASE "-O2")
# set(CMAKE_CXX_FLAGS "${CMAKE_C_FLAGS} -lstdc++")
set(CMAKE_CXX_FLAGS_DEBUG ${CMAKE_C_FLAGS_DEBUG})
set(CMAKE_CXX_FLAGS_RELEASE ${CMAKE_C_FLAGS_RELEASE})

#set (CMAKE_LIBRARY_PATH "C://Program Files//NVIDIA GPU Computing Toolkit//CUDA//v11.6//lib//")

set(CMAKE_BUILD_TYPE release)
# set(CMAKE_BUILD_TYPE debug)

# Find OpenCV, you may need to set OpenCV_DIR variable
# to the absolute path to the directory containing OpenCVConfig.cmake file
# via the command line or GUI
SET("OpenCV_DIR" "C:\\opencv\\build_455")
find_package(OpenCV REQUIRED)

# set(THREADS_PREFER_PTHREAD_FLAG ON)
#find_package(Threads REQUIRED)

# If the package has been found, several variables will
# be set, you can find the full list with descriptions
# in the OpenCVConfig.cmake file.
# Print some message showing some of them
message(STATUS "OpenCV library status:")
message(STATUS "    config: ${OpenCV_DIR}")
message(STATUS "    version: ${OpenCV_VERSION}")
message(STATUS "    libraries: ${OpenCV_LIBS}")
message(STATUS "    include path: ${OpenCV_INCLUDE_DIRS}")
include_directories( ${OpenCV_INCLUDE_DIRS} )

find_package(CUDA REQUIRED)
message(STATUS "CUDA library status:")
message(STATUS "    config: ${CUDA_DIR}")
message(STATUS "    version: ${CUDA_VERSION}")
message(STATUS "    libraries: ${CUDA_LIBRARIES}")
message(STATUS "    include path: ${CUDA_INCLUDE_DIRS}")
include_directories( ${CUDA_INCLUDE_DIRS})

# Declare the executable target built from your sources
# add_executable(trt_efficientdet main_image.cpp object_detector.cpp ${CMAKE_SOURCE_DIR}/../../TensorRT/samples/common/logger.cpp)
#add_executable(trt_efficientdet main.cpp  inference.cpp dasiamrpn_kernel_r1.cpp ${CMAKE_SOURCE_DIR}/../../TensorRT/samples/common/logger.cpp)

add_executable(trt_efficientdet tracker_dasiamrpn.cpp main.cpp inference_fp16.cpp C://Users//biaspaltsau_aa//Downloads//TensorRT-8.2.1.8.Windows10.x86_64.cuda-11.4.cudnn8.2//TensorRT-8.2.1.8//samples//common//logger.cpp)

# Link your application with OpenCV libraries
target_link_libraries(trt_efficientdet LINK_PRIVATE ${OpenCV_LIBS})
target_link_libraries(trt_efficientdet LINK_PRIVATE ${CUDA_LIBRARIES})


EXECUTE_PROCESS(COMMAND uname -m COMMAND tr -d '\n' OUTPUT_VARIABLE ARCHITECTURE)
message(STATUS "Architecture: ${ARCHITECTURE}")

# target_link_libraries(trt_efficientdet Threads::Threads)

find_library(LIBRT rt)
if(LIBRT)
  target_link_libraries(trt_efficientdet LINK_PRIVATE ${LIBRT})
endif()

find_library(LIBDL dl)
if(LIBDL)
  target_link_libraries(trt_efficientdet LINK_PRIVATE ${LIBDL})
endif()

target_link_libraries(trt_efficientdet LINK_PRIVATE cudnn)
target_link_libraries(trt_efficientdet LINK_PRIVATE nvinfer)
target_link_libraries(trt_efficientdet LINK_PRIVATE nvinfer_plugin)
#add_library(trt_efficientdet "C://Program Files//NVIDIA GPU Computing Toolkit//CUDA//v11.6//lib//cudnn.lib")
target_link_libraries(trt_efficientdet LINK_PRIVATE "C://Program Files//NVIDIA GPU Computing Toolkit//CUDA//v11.6//lib//cudnn.lib")

target_include_directories(trt_efficientdet PUBLIC ${CMAKE_SOURCE_DIR})
target_include_directories(trt_efficientdet PUBLIC C://Users//biaspaltsau_aa//Downloads//TensorRT-8.2.1.8.Windows10.x86_64.cuda-11.4.cudnn8.2//TensorRT-8.2.1.8//include)
target_include_directories(trt_efficientdet PUBLIC C://Users//biaspaltsau_aa//Downloads//TensorRT-8.2.1.8.Windows10.x86_64.cuda-11.4.cudnn8.2//TensorRT-8.2.1.8//samples//common)
target_include_directories(trt_efficientdet PUBLIC "C://Program Files//NVIDIA GPU Computing Toolkit//CUDA//v11.6//include")
target_include_directories(trt_efficientdet PUBLIC "C://Program Files//NVIDIA GPU Computing Toolkit//CUDA//v11.6//lib")



if (MSVC)
  file(GLOB OpenCV_DLLS "${OpenCV_DIR}/bin/Release/*.dll")
  add_custom_command(TARGET trt_efficientdet
                     POST_BUILD
                     COMMAND ${CMAKE_COMMAND} -E copy_if_different
                     ${OpenCV_DLLS}
                     $<TARGET_FILE_DIR:trt_efficientdet>)

endif (MSVC)
