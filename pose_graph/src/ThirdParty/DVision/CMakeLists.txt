find_package(OpenCV REQUIRED)
# add Boost for dynamic_bitset
find_package(Boost REQUIRED COMPONENTS system filesystem)

add_library(DVision STATIC BRIEF.cpp)
target_link_libraries(DVision ${OpenCV_LIBS} ${Boost_LIBRARIES})
target_include_directories(DVision PUBLIC 
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
    $<INSTALL_INTERFACE:include>
    PRIVATE 
    ${OpenCV_INCLUDE_DIRS}
    ${Boost_INCLUDE_DIRS}
    )