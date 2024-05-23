find_package(OpenCV REQUIRED)

add_library(DBoW STATIC
    BowVector.cpp
    FBrief.cpp
    FeatureVector.cpp
    QueryResults.cpp
    ScoringObject.cpp
    )
target_link_libraries(DBoW 
    ${OpenCV_LIBS} 
    DVision
    DUtils
    VocabularyBinary
    )
target_include_directories(DBoW PUBLIC 
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
    $<INSTALL_INTERFACE:include>
    PRIVATE 
    ${OpenCV_INCLUDE_DIRS}
    )