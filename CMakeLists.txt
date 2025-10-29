cmake_minimum_required(VERSION 3.10)
project(vector_search)
set(CMAKE_CXX_STANDARD 17)

add_library(vector_search SHARED src/vector_search.cpp)

find_package(OpenMP)
find_package(Python3 COMPONENTS Interpreter Development REQUIRED)
find_package(pybind11 REQUIRED)

pybind11_add_module(vector_search src/vector_search.cpp src/bindings/pybind_module.cpp)
if (OpenMP_CXX_FOUND)
    target_link_libraries(vector_search PUBLIC OpenMP::OpenMP_CXX)
endif()