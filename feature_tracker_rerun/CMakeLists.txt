cmake_minimum_required(VERSION 3.0)
project(feature_tracker_rerun VERSION 1.0)
set(CMAKE_CXX_FLAGS "-std=c++17")
include(FetchContent)
FetchContent_Declare(rerun_sdk URL
    https://github.com/rerun-io/rerun/releases/latest/download/rerun_cpp_sdk.zip)
FetchContent_MakeAvailable(rerun_sdk)

find_package(OpenCV REQUIRED)
find_package(feature_tracker REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(spdlog REQUIRED)
find_package(yaml-cpp REQUIRED)

# Define your library target
add_library(feature_tracker_rerun STATIC src/feature_tracker_observer_spdlog_rerun.cpp)
target_link_libraries(feature_tracker_rerun PRIVATE ${OpenCV_LIBS} spdlog::spdlog_header_only rerun_sdk feature_tracker::feature_tracker camodocal::camodocal yaml-cpp)

# Set include directories
target_include_directories(feature_tracker_rerun
    PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

add_executable(feature_tracker_run
    src/main.cpp

)
target_link_libraries(feature_tracker_run feature_tracker_rerun feature_tracker ${OpenCV_LIBS})
# Install the library
include(GNUInstallDirs)
install(TARGETS feature_tracker_rerun
    EXPORT feature_tracker_rerunTargets
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

# Install headers
install(DIRECTORY include/
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

# Generate and install CMake package configuration files
install(EXPORT feature_tracker_rerunTargets
    FILE feature_tracker_rerunTargets.cmake
    NAMESPACE feature_tracker_rerun::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/feature_tracker_rerun
)


# Generate and install CMake package configuration files
include(CMakePackageConfigHelpers)
write_basic_package_version_file(
    ${CMAKE_CURRENT_BINARY_DIR}/feature_tracker_rerunConfigVersion.cmake
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY SameMajorVersion
)

configure_package_config_file(
    ${CMAKE_CURRENT_LIST_DIR}/cmake/feature_tracker_rerunConfig.cmake.in
    ${CMAKE_CURRENT_BINARY_DIR}/feature_tracker_rerunConfig.cmake
    INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/feature_tracker_rerun
)

install(FILES
    ${CMAKE_CURRENT_BINARY_DIR}/feature_tracker_rerunConfig.cmake
    ${CMAKE_CURRENT_BINARY_DIR}/feature_tracker_rerunConfigVersion.cmake
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/feature_tracker_rerun
)
