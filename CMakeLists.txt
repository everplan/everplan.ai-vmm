cmake_minimum_required(VERSION 3.16)
project(ai-vmm VERSION 0.1.0)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Build options
option(BUILD_TESTS "Build tests" ON)
option(BUILD_PYTHON_BINDINGS "Build Python bindings" ON)
option(ENABLE_INTEL_BACKEND "Enable Intel hardware backend" ON)
option(ENABLE_NVIDIA_BACKEND "Enable NVIDIA CUDA backend" ON)
option(ENABLE_AMD_BACKEND "Enable AMD ROCm backend" OFF)

# Find packages
find_package(PkgConfig REQUIRED)

# Intel backends
if(ENABLE_INTEL_BACKEND)
    # OpenVINO for NPU and optimization
    find_package(OpenVINO QUIET)
    if(OpenVINO_FOUND)
        message(STATUS "OpenVINO found - Intel NPU support enabled")
        add_compile_definitions(AI_VMM_OPENVINO_AVAILABLE)
    else()
        message(WARNING "OpenVINO not found - Intel NPU support disabled")
    endif()
    
    # oneAPI components (optional but recommended)
    find_package(oneDNN CONFIG QUIET)
    find_package(TBB CONFIG QUIET)
    
    if(oneDNN_FOUND)
        message(STATUS "oneDNN found - Intel CPU optimizations enabled")
        add_compile_definitions(AI_VMM_ONEDNN_AVAILABLE)
    endif()
    
    if(TBB_FOUND)
        message(STATUS "TBB found - Intel threading optimizations enabled")
        add_compile_definitions(AI_VMM_TBB_AVAILABLE)
    endif()
    
    add_compile_definitions(AI_VMM_INTEL_BACKEND)
endif()

# NVIDIA backend
if(ENABLE_NVIDIA_BACKEND)
    find_package(CUDA QUIET)
    if(CUDA_FOUND)
        enable_language(CUDA)
        find_package(CUDAToolkit QUIET)
        if(CUDAToolkit_FOUND)
            message(STATUS "CUDA found - NVIDIA GPU support enabled")
            add_compile_definitions(AI_VMM_NVIDIA_BACKEND)
        else()
            message(WARNING "CUDA toolkit not found - NVIDIA backend disabled")
            set(ENABLE_NVIDIA_BACKEND OFF)
        endif()
    else()
        message(WARNING "CUDA not found - NVIDIA backend disabled")
        set(ENABLE_NVIDIA_BACKEND OFF)
    endif()
endif()

# AMD backend (future)
if(ENABLE_AMD_BACKEND)
    # TODO: Add ROCm detection
    add_compile_definitions(AI_VMM_AMD_BACKEND)
endif()

# Python bindings
if(BUILD_PYTHON_BINDINGS)
    find_package(pybind11 QUIET)
    if(pybind11_FOUND)
        message(STATUS "pybind11 found - Python bindings enabled")
    else()
        message(WARNING "pybind11 not found - Python bindings disabled")
        set(BUILD_PYTHON_BINDINGS OFF)
    endif()
endif()

# Include directories
include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)

# Core library
add_subdirectory(src/core)
add_subdirectory(src/hal)
add_subdirectory(src/optimization)
add_subdirectory(src/scheduling)
add_subdirectory(src/memory)

# Backend implementations (unified)
add_subdirectory(src/backends)

# Python bindings
if(BUILD_PYTHON_BINDINGS AND EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/src/python")
    add_subdirectory(src/python)
endif()

# Tests
if(BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()

# Main library target
add_library(ai-vmm STATIC
    $<TARGET_OBJECTS:ai_vmm_core>
    $<TARGET_OBJECTS:ai_vmm_hal>
    $<TARGET_OBJECTS:ai_vmm_optimization>
    $<TARGET_OBJECTS:ai_vmm_scheduling>
    $<TARGET_OBJECTS:ai_vmm_memory>
)

target_link_libraries(ai-vmm PUBLIC
    ai_vmm_core
    ai_vmm_hal
    ai_vmm_optimization
    ai_vmm_scheduling
    ai_vmm_memory
)

# Examples
add_subdirectory(examples)

# Installation
install(TARGETS ai-vmm
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
    RUNTIME DESTINATION bin
)

install(DIRECTORY include/ DESTINATION include)

# Package configuration
include(CMakePackageConfigHelpers)
write_basic_package_version_file(
    "${CMAKE_CURRENT_BINARY_DIR}/ai-vmm-config-version.cmake"
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY SameMajorVersion
)