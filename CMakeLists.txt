cmake_minimum_required(VERSION 3.9)
project(cuda_mat)

set(CMAKE_CXX_STANDARD 17)

SET(GCC_COVERAGE_COMPILE_FLAGS "-fconcepts")
#SET(GCC_COVERAGE_LINK_FLAGS    "")


add_definitions(${GCC_COVERAGE_COMPILE_FLAGS})

add_executable(cuda_mat main.cpp Matrix.h util.h givens.h)