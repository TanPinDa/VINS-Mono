project(pose_graph_sample)
cmake_minimum_required(VERSION 3.5)

set(CMAKE_CXX_STANDARD 17)

find_package(pose_graph REQUIRED)

add_executable(pose_graph_sample main.cpp)
target_link_libraries(pose_graph_sample pose_graph::pose_graph)

