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

find_package(OpenCV)
find_package(camodocal REQUIRED)
# find_package(Ceres REQUIRED) # TODO Remove because it causes failure due to multiple GLOG. 

set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)
find_package(Eigen3)

# include_directories( ${CERES_INCLUDE_DIRS}  ${EIGEN3_INCLUDE_DIR})

catkin_package()

add_executable(pose_graph_node
    src/pose_graph_node.cpp
    )

target_link_libraries(pose_graph_node pose_graph ${catkin_LIBRARIES} camodocal::camodocal)
# target_include_directories(pose_graph_node PUBLIC
#     ${catkin_INCLUDE_DIRS}
# )
# message("catkin_lib  ${catkin_LIBRARIES}")
