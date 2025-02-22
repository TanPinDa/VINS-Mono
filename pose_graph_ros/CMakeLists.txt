cmake_minimum_required(VERSION 2.8.3)
project(pose_graph_ros)

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

find_package(OpenCV REQUIRED)
find_package(pose_graph REQUIRED)
find_package(Eigen3 REQUIRED)

catkin_package()

add_executable(pose_graph_node
    src/pose_graph_node.cpp
    src/utility/CameraPoseVisualization.cpp
    )

target_link_libraries(pose_graph_node pose_graph::pose_graph ${catkin_LIBRARIES})
target_include_directories(pose_graph_node PUBLIC
    ${catkin_INCLUDE_DIRS}
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)
