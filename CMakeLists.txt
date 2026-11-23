cmake_minimum_required(VERSION 3.18)
project(VCORE_L1_CERTIFIED LANGUAGES CXX CUDA)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CXX_STANDARD 17)

find_package(CUDAToolkit REQUIRED)

add_library(vcore_engine SHARED
    cuda/VCoreEngine.cpp
    cuda/vcore_quantize_kernel.cu
)

set_target_properties(vcore_engine PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
target_link_libraries(vcore_engine PUBLIC CUDA::cudart)
