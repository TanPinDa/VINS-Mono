cmake_minimum_required(VERSION 2.8.3)
project(pose_graph)

set(CMAKE_BUILD_TYPE "Release")
set(CMAKE_CXX_FLAGS "-std=c++17")
#-DEIGEN_USE_MKL_ALL")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -Wall -g")

find_package(catkin REQUIRED COMPONENTS
    roscpp
    std_msgs
    nav_msgs
    cv_bridge
    roslib
    )

find_package(OpenCV)
find_package(camodocal REQUIRED)
# find_package(Ceres REQUIRED) # TODO Remove because it causes failure due to multiple GLOG. 

set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)
find_package(Eigen3)

# include_directories( ${CERES_INCLUDE_DIRS}  ${EIGEN3_INCLUDE_DIR})

catkin_package()

add_subdirectory(src/ThirdParty)

add_library(pose_graph STATIC
    src/pose_graph.cpp
    src/keyframe.cpp
    src/utility/CameraPoseVisualization.cpp
    )
target_link_libraries(pose_graph 
    DBoW
    DUtils
    DVision
    VocabularyBinary
    ${catkin_LIBRARIES} 
    ${OpenCV_LIBS} 
    ${CERES_LIBRARIES}
    )
target_include_directories(pose_graph PUBLIC 
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
    ${OpenCV_INCLUDE_DIRS} 
    ${CERES_INCLUDE_DIRS}
    ${catkin_INCLUDE_DIRS}
    )

add_executable(pose_graph_node
    src/pose_graph_node.cpp
    )


target_link_libraries(pose_graph_node pose_graph camodocal)
# message("catkin_lib  ${catkin_LIBRARIES}")
