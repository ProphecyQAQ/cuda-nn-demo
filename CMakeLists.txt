cmake_minimum_required(VERSION 3.24)

# Make sure CMake can find CUDA even when /usr/local/cuda/bin is not on PATH.
if (NOT DEFINED CMAKE_CUDA_COMPILER AND NOT DEFINED ENV{CUDACXX})
    find_program(_DEMO_NVCC nvcc HINTS /usr/local/cuda/bin)
    if (_DEMO_NVCC)
        set(CMAKE_CUDA_COMPILER "${_DEMO_NVCC}" CACHE FILEPATH "CUDA compiler")
    endif()
endif()

project(tiny-cuda-nn-demo
    VERSION 1.0.0
    DESCRIPTION "Fit a 2D image with tiny-cuda-nn (hash grid + MLP)"
    LANGUAGES CXX CUDA)

# ---------------------------------------------------------------------------
# Global settings
# ---------------------------------------------------------------------------
if (NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
    set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS Debug Release RelWithDebInfo MinSizeRel)
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_EXTENSIONS OFF)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# ---------------------------------------------------------------------------
# Third-party: tiny-cuda-nn (build from the pinned clone in third_party/)
# ---------------------------------------------------------------------------
set(TCNN_DIR "${CMAKE_CURRENT_SOURCE_DIR}/third_party/tiny-cuda-nn")

if (NOT EXISTS "${TCNN_DIR}/CMakeLists.txt")
    message(FATAL_ERROR
        "tiny-cuda-nn was not found at ${TCNN_DIR}.\n"
        "Run ./scripts/setup_deps.sh first (or clone it with --recursive into third_party/).")
endif()
if (NOT EXISTS "${TCNN_DIR}/dependencies/cutlass/CMakeLists.txt")
    message(FATAL_ERROR
        "tiny-cuda-nn submodules are missing.\n"
        "Run ./scripts/setup_deps.sh, or: git -C third_party/tiny-cuda-nn submodule update --init --recursive")
endif()

# Only the library is needed here; its examples/benchmarks/tests would slow the build down.
set(TCNN_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(TCNN_BUILD_BENCHMARK OFF CACHE BOOL "" FORCE)
set(TCNN_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(TCNN_BUILD_WITH_RTC ON CACHE BOOL "" FORCE)

# tiny-cuda-nn auto-detects the GPU architecture, or honours -DTCNN_CUDA_ARCHITECTURES=...
add_subdirectory("${TCNN_DIR}" EXCLUDE_FROM_ALL)

# tiny-cuda-nn propagates the architecture it selected through TCNN_CUDA_ARCHITECTURES
# (a parent-scope variable). Our own kernels must target the same architecture.
if (TCNN_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES "${TCNN_CUDA_ARCHITECTURES}")
endif()

# ---------------------------------------------------------------------------
# image I/O (stb_image + stb_image_write)
# ---------------------------------------------------------------------------
add_library(demo_image_io STATIC src/image_io.cpp)
target_include_directories(demo_image_io PUBLIC
    "${CMAKE_CURRENT_SOURCE_DIR}/src"
    "${CMAKE_CURRENT_SOURCE_DIR}/third_party/stb")

if (CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(demo_image_io PRIVATE -Wall -Wextra)
endif()

# ---------------------------------------------------------------------------
# Demo executable
# ---------------------------------------------------------------------------
file(TO_CMAKE_PATH "${CMAKE_CURRENT_SOURCE_DIR}" DEMO_SOURCE_DIR)
configure_file(src/demo_config.h.in "${CMAKE_CURRENT_BINARY_DIR}/generated/demo_config.h" @ONLY)

add_executable(tcnn_demo src/tcnn_demo.cu)

target_include_directories(tcnn_demo PRIVATE "${CMAKE_CURRENT_BINARY_DIR}/generated")
target_link_libraries(tcnn_demo PRIVATE tiny-cuda-nn demo_image_io)

set_target_properties(tcnn_demo PROPERTIES
    CUDA_SEPARABLE_COMPILATION OFF
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
message(STATUS "")
message(STATUS "tiny-cuda-nn-demo")
message(STATUS "  build type         : ${CMAKE_BUILD_TYPE}")
message(STATUS "  CUDA architectures : ${CMAKE_CUDA_ARCHITECTURES}")
message(STATUS "  tiny-cuda-nn       : ${TCNN_DIR}")
message(STATUS "  executable         : ${CMAKE_BINARY_DIR}/tcnn_demo")
message(STATUS "")
