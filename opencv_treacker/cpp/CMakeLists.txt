cmake_minimum_required(VERSION 3.5)




project( cv2_u )

find_package(OpenCV 4 QUIET)
if (NOT OpenCV_FOUND)
  find_package(OpenCV REQUIRED)
endif ()
message(STATUS "Found OpenCV version ${OpenCV_VERSION}")





find_package(Threads REQUIRED)
#find_package(Boost COMPONENTS unit_test_framework REQUIRED)



include_directories( ${OpenCV_INCLUDE_DIRS} )

set(SOURCES 
		"main.cpp"
	
		)

set(HEADERS

		
		)
		
add_executable(cv2_u ${SOURCES} ${HEADERS} "main.cpp")

target_link_libraries( cv2_u ${OpenCV_LIBS} )

set_target_properties(cv2_u PROPERTIES
						CXX_STANDARD 11
						CXX_STANDARD_REQUIRED ON)