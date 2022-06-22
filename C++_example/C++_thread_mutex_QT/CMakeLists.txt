# CMakeList.txt : Top-level CMake project file, do global configuration
# and include sub-projects here.
#
cmake_minimum_required (VERSION 3.10.0)

project ("CMake_Torch")


# Include sub-projects.
add_subdirectory ("CMake_Torch")

add_subdirectory ("QT")
