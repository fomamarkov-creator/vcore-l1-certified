cmake_minimum_required(VERSION 3.18)
project(VCORE_L1_CERTIFIED LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 1. Нативный CPU-движок (AVX2 / SIMD без распаковки в float32)
add_library(vcore_cpu SHARED cpp/vcore_cpu_engine.cpp)
if(CMAKE_SYSTEM_PROCESSOR MATCHES "(x86_64|AMD64)")
    target_compile_options(vcore_cpu PRIVATE -O3 -mavx2)
else()
    target_compile_options(vcore_cpu PRIVATE -O3)
endif()

# 2. Нативный GPU CUDA-движок (если установлен CUDA Toolkit)
include(CheckLanguage)
check_language(CUDA)
if(CMAKE_CUDA_COMPILER)
    enable_language(CUDA)
    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_ARCHITECTURES "75;80;86;89;90")

    find_package(CUDAToolkit REQUIRED)

    add_library(vcore_engine SHARED
        cuda/VCoreEngine.cpp
        cuda/vcore_quantize_kernel.cu
    )

    set_target_properties(vcore_engine PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
    target_link_libraries(vcore_engine PUBLIC CUDA::cudart)
endif()
# 3. Автономный CLI-бинарник vcore-run (чистый C++17 + AVX2, без Python/PyTorch)
add_executable(vcore-run cpp/vcore_run.cpp)
target_link_libraries(vcore-run PRIVATE vcore_cpu)
