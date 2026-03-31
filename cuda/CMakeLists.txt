cmake_minimum_required(VERSION 3.18)
project(kvtc_cuda LANGUAGES CXX CUDA)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CXX_STANDARD 17)

# Auto-detect GPU architecture
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES native)
endif()

# KVTC kernel library
add_library(kvtc_kernels STATIC
    kvtc_kernels.cu
)
target_include_directories(kvtc_kernels PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

# Test executable
add_executable(test_kvtc
    test_kvtc_kernels.cu
    kvtc_kernels.cu
)
target_include_directories(test_kvtc PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(test_kvtc PRIVATE ${CUDA_LIBRARIES})

# Build instructions:
# mkdir build && cd build
# cmake -G Ninja ..
# cmake --build . --target test_kvtc
# ./test_kvtc
