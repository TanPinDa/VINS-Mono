add_library(DUtils STATIC
    Random.cpp
    Timestamp.cpp
    )
target_include_directories(DUtils PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
    $<INSTALL_INTERFACE:include>
    )