cmake_minimum_required(VERSION 3.18)
project(vcore_l1_engine LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenMP)

# 1. Кроссплатформенный движок ARM NEON + Sub-1-Bit E8 LUT (x86_64, ARM64, macOS, Android)
add_library(vcore_lut_neon SHARED cpp/vcore_lut_neon.cpp)
target_compile_options(vcore_lut_neon PRIVATE -O3)
if(OpenMP_CXX_FOUND)
    target_link_libraries(vcore_lut_neon PRIVATE OpenMP::OpenMP_CXX)
endif()

# 2. x86_64 AVX2 движок и автономный раннер vcore-run
if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64|amd64")
    add_library(vcore_cpu SHARED cpp/vcore_cpu_engine.cpp)
    target_compile_options(vcore_cpu PRIVATE -O3 -mavx2 -mfma)

    add_executable(vcore-run cpp/vcore_run.cpp cpp/vcore_cpu_engine.cpp)
    target_compile_options(vcore-run PRIVATE -O3 -mavx2 -mfma)
    if(OpenMP_CXX_FOUND)
        target_link_libraries(vcore_cpu PRIVATE OpenMP::OpenMP_CXX)
        target_link_libraries(vcore-run PRIVATE OpenMP::OpenMP_CXX)
    endif()
else()
    # На ARM64 (Apple Silicon / Android Termux) собираем vcore-run без x86 флагов
    add_executable(vcore-run cpp/vcore_run.cpp cpp/vcore_cpu_engine.cpp)
    target_compile_options(vcore-run PRIVATE -O3)
endif()

# 3. Опциональный CUDA движок (если найден nvcc)
include(CheckLanguage)
check_language(CUDA)
if(CMAKE_CUDA_COMPILER)
    enable_language(CUDA)
    set(CMAKE_CUDA_STANDARD 17)
    add_library(vcore_engine SHARED
        cuda/VCoreEngine.cpp
        cuda/vcore_quantize_kernel.cu
    )
    set_target_properties(vcore_engine PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_ARCHITECTURES "75;80;86;89;90"
    )
endif()
