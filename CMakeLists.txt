cmake_minimum_required(VERSION 3.9)
project(cuda_mat)
 
set(CMAKE_CXX_STANDARD 14)
 
#SET(GCC_COVERAGE_COMPILE_FLAGS "-fconcepts") required for Matrix.h and util.h
#SET(GCC_COVERAGE_LINK_FLAGS    "")
 
find_package(CUDA QUIET REQUIRED)
find_package(OpenMP REQUIRED)
#set(CUDA_VERBOSE_BUILD ON)
 
#add_executable(dummy dummy.cpp) #needed to bypass clion-cuda issue
 
 
#add_executable(cuda_mat main.cpp Matrix.h util.h givens.h) //wont work with gcc < 7 (cuda requires it)
 
link_directories(${CUDA_TOOLKIT_ROOT_DIR}/lib64)
link_directories(${CUDA_TOOLKIT_ROOT_DIR}/lib/x64)

include_directories(sys)
 
cuda_add_executable(
        cuda
        main.cu helper_math.h sys/times.h sys/times.cpp)
 
 
target_link_libraries(cuda cusparse cusolver ${OpenMP_CXX_FLAGS})