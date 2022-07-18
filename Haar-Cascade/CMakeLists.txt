# CMakeList.txt : CMake project for Project, include source and define
# project specific logic here.
#
cmake_minimum_required (VERSION 3.8)

project ("Project")

find_package( OpenCV REQUIRED )

include_directories( ${OpenCV_INCLUDE_DIRS} )

# Add source to this project's executable.
add_executable (Project "Project.cpp" "includes/header.h" "libs/utilities.cpp" "includes/tinyxml2.h" "libs/tinyxml2.cpp")

target_link_libraries( ${PROJECT_NAME} ${OpenCV_LIBS} )
