cmake_minimum_required(VERSION 3.9)
project(cuda_mat)
 
set(CMAKE_CXX_STANDARD 14)
 
#SET(GCC_COVERAGE_COMPILE_FLAGS "-fconcepts") required for Matrix.h and util.h
#SET(GCC_COVERAGE_LINK_FLAGS    "")
 
find_package(CUDA QUIET REQUIRED)
#set(CUDA_VERBOSE_BUILD ON)

set(CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS ON)
 
#add_executable(dummy dummy.cpp) #needed to bypass clion-cuda issue
 
 
#add_executable(cuda_mat main.cpp Matrix.h util.h givens.h) //wont work with gcc < 7 (cuda requires it)

 
link_directories(${CUDA_TOOLKIT_ROOT_DIR}/lib64)
link_directories(${CUDA_TOOLKIT_ROOT_DIR}/lib/x64)
include_directories(${CUDA_TOOLKIT_ROOT_DIR}/include)
 


cuda_add_library(cuda_mat SHARED mmio.c mmio.h mmio_wrapper.h helper_cuda.h helper_cusolver.h helper_string.h pbicgstab.h pbicgstab.cpp)

target_link_libraries(cuda_mat cusparse cublas cudart_static)
add_executable(example pbicgstab.h example.cpp)

target_link_libraries(example cuda_mat)
