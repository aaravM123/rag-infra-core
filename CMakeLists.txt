cmake_minimum_required(VERSION 3.10)
project(vector_search)
set(CMAKE_CXX_STANDARD 17)

find_package(OpenMP)
find_package(Python3 COMPONENTS Interpreter Development REQUIRED)

# Help CMake locate pybind11 when installed via pip in a virtualenv
execute_process(
    COMMAND "${Python3_EXECUTABLE}" -c "import pybind11, pathlib; print(pybind11.get_cmake_dir())"
    OUTPUT_VARIABLE PYBIND11_CMAKE_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
    ERROR_QUIET
)
if (PYBIND11_CMAKE_DIR AND EXISTS "${PYBIND11_CMAKE_DIR}")
    list(APPEND CMAKE_PREFIX_PATH "${PYBIND11_CMAKE_DIR}")
endif()

find_package(pybind11 REQUIRED)

pybind11_add_module(vector_search src/bindings/pybind_module.cpp)

if (OpenMP_CXX_FOUND)
    target_link_libraries(vector_search PUBLIC OpenMP::OpenMP_CXX)
endif()