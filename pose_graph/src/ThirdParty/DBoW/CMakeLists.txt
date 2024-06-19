find_package(OpenCV REQUIRED)

add_library(DBoW STATIC
    src/BowVector.cpp
    src/FBrief.cpp
    src/FeatureVector.cpp
    src/QueryResults.cpp
    src/ScoringObject.cpp
    )
target_link_libraries(DBoW 
    ${OpenCV_LIBS} 
    DVision
    DUtils
    VocabularyBinary
    )
target_include_directories(DBoW PUBLIC 
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
    PRIVATE 
    ${OpenCV_INCLUDE_DIRS}
    )