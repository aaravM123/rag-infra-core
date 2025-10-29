cmake_minimum_required(VERSION 3.10)
project(vector_search)
set(CMAKE_CXX_STANDARD 17)

add_library(vector_search SHARED src/vector_search.cpp)

find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    target_link_libraries(vector_search PUBLIC OpenMP::OpenMP_CXX)
endif()

# Output to build/lib
set_target_properties(vector_search PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
)
