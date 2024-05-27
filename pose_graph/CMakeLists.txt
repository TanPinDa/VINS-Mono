cmake_minimum_required(VERSION 2.8.3)
project(pose_graph)

set(CMAKE_BUILD_TYPE "Release")
set(CMAKE_CXX_FLAGS "-std=c++17")
#-DEIGEN_USE_MKL_ALL")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -Wall -g")

find_package(OpenCV)
find_package(Ceres REQUIRED)

set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)
find_package(Eigen3)

add_subdirectory(src/ThirdParty)

add_library(pose_graph STATIC
    src/pose_graph.cpp
    src/keyframe.cpp
    )
target_link_libraries(pose_graph PUBLIC
    ${OpenCV_LIBS}
    PRIVATE 
    DBoW
    DUtils
    DVision
    VocabularyBinary
    Ceres::ceres
    )
target_include_directories(pose_graph PUBLIC 
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
    ${OpenCV_INCLUDE_DIRS}
    )
