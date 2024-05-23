find_package(OpenCV REQUIRED)

add_library(VocabularyBinary VocabularyBinary.cpp)
target_link_libraries(VocabularyBinary ${OpenCV_LIBS})
target_include_directories(VocabularyBinary PUBLIC 
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
    $<INSTALL_INTERFACE:include>
    PRIVATE 
    ${OpenCV_INCLUDE_DIRS}
    )