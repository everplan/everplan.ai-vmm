# AI VMM Backends Library

# Backend interface implementation
set(BACKEND_SOURCES
    compute_backend.cpp
    backend_manager.cpp
)

# Intel backend (always available since it supports CPU)
list(APPEND BACKEND_SOURCES intel_backend.cpp)

# NVIDIA backend (optional)
if(BUILD_NVIDIA_BACKEND)
    list(APPEND BACKEND_SOURCES nvidia_backend.cpp)
endif()

# AMD backend (optional - placeholder for future)
if(BUILD_AMD_BACKEND)
    # list(APPEND BACKEND_SOURCES amd_backend.cpp)
endif()

# Create the backends library
add_library(ai_vmm_backends STATIC ${BACKEND_SOURCES})

# Set target properties
set_target_properties(ai_vmm_backends PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
    POSITION_INDEPENDENT_CODE ON
)

# Include directories
target_include_directories(ai_vmm_backends
    PUBLIC
        ${CMAKE_SOURCE_DIR}/include
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
)

# Link dependencies
target_link_libraries(ai_vmm_backends
    PUBLIC
        # Remove the circular dependency to core
    PRIVATE
        ${CMAKE_THREAD_LIBS_INIT}
)

# Intel backend dependencies
if(BUILD_INTEL_BACKEND)
    if(TARGET Intel::SYCL)
        target_link_libraries(ai_vmm_backends PRIVATE Intel::SYCL)
        target_compile_definitions(ai_vmm_backends PRIVATE AI_VMM_HAS_ONEAPI)
    endif()
    
    if(TARGET OpenVINO::openvino)
        target_link_libraries(ai_vmm_backends PRIVATE OpenVINO::openvino)
        target_compile_definitions(ai_vmm_backends PRIVATE AI_VMM_HAS_OPENVINO)
    endif()
endif()

# NVIDIA backend dependencies
if(BUILD_NVIDIA_BACKEND)
    if(TARGET CUDA::cudart)
        target_link_libraries(ai_vmm_backends PRIVATE CUDA::cudart CUDA::cuda_driver)
        target_compile_definitions(ai_vmm_backends PRIVATE AI_VMM_HAS_CUDA)
    endif()
    
    if(TARGET CUDA::curand)
        target_link_libraries(ai_vmm_backends PRIVATE CUDA::curand)
    endif()
    
    if(TARGET CUDA::cublas)
        target_link_libraries(ai_vmm_backends PRIVATE CUDA::cublas)
    endif()
    
    if(TARGET CUDA::cudnn)
        target_link_libraries(ai_vmm_backends PRIVATE CUDA::cudnn)
        target_compile_definitions(ai_vmm_backends PRIVATE AI_VMM_HAS_CUDNN)
    endif()
endif()

# AMD backend dependencies
if(BUILD_AMD_BACKEND)
    # Future: ROCm dependencies
endif()

# Compiler-specific options
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(ai_vmm_backends PRIVATE
        -Wall -Wextra -Wpedantic
        -Wno-unused-parameter
        -fvisibility=hidden
    )
endif()

if(MSVC)
    target_compile_options(ai_vmm_backends PRIVATE
        /W4 /WX- /permissive-
    )
endif()

# Installation
if(AI_VMM_INSTALL)
    install(TARGETS ai_vmm_backends
        EXPORT ai_vmm_targets
        ARCHIVE DESTINATION lib
        LIBRARY DESTINATION lib
        RUNTIME DESTINATION bin
    )
endif()