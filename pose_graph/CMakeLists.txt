cmake_minimum_required(VERSION 3.0)
project(pose_graph VERSION 1.0)

set(CMAKE_BUILD_TYPE "Release")
set(CMAKE_CXX_FLAGS "-std=c++17")

# ------------ Find dependencies ------------
find_package(OpenCV REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(camodocal REQUIRED)
# ------------------------------------------

# --- Search for third party dependencies ---
add_subdirectory(src/ThirdParty)
# ------------------------------------------

# -------- Add library --------
add_library(pose_graph STATIC
    src/pose_graph.cpp
    src/keyframe.cpp
    )
target_link_libraries(pose_graph PUBLIC
    ${OpenCV_LIBS}
    ${EIGEN3_LIBS}
    DBoW
    DUtils
    DVision
    VocabularyBinary
    camodocal::camodocal 
    )
target_include_directories(pose_graph PUBLIC 
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
    ${OpenCV_INCLUDE_DIRS}
    ${EIGEN3_INCLUDE_DIR}
    )

# -----------------------------

# ----------- Install -----------
include(GNUInstallDirs)

install(TARGETS pose_graph DBoW DUtils DVision VocabularyBinary
    EXPORT pose_graphTargets
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
    )

install(DIRECTORY include/ 
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
    )

install (DIRECTORY src/ThirdParty/DBoW/include/
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
    )
install(DIRECTORY src/ThirdParty/DUtils/
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
    )
install(DIRECTORY src/ThirdParty/DVision/
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
    )
install(DIRECTORY src/ThirdParty/VocabularyBinary/
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
    )

install(EXPORT pose_graphTargets
    FILE pose_graphTargets.cmake
    NAMESPACE pose_graph::
    DESTINATION lib/cmake/pose_graph
    )

include(CMakePackageConfigHelpers)
write_basic_package_version_file(
    "${CMAKE_CURRENT_BINARY_DIR}/pose_graphConfigVersion.cmake"
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY SameMajorVersion
    )

configure_package_config_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/cmake/pose_graphConfig.cmake.in
    ${CMAKE_CURRENT_BINARY_DIR}/pose_graphConfig.cmake
    INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/pose_graph
    )

install(FILES
    ${CMAKE_CURRENT_BINARY_DIR}/pose_graphConfig.cmake
    ${CMAKE_CURRENT_BINARY_DIR}/pose_graphConfigVersion.cmake
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/pose_graph
    )
# -----------------------------