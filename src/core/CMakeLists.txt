add_library(ai_vmm_core OBJECT
    vmm.cpp
    model.cpp
    hardware_discovery.cpp
)

target_include_directories(ai_vmm_core PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/../../include
)

# Link to backends for the new backend manager
target_link_libraries(ai_vmm_core PRIVATE ai_vmm_backends)

# Platform-specific linking
if(WIN32)
    target_link_libraries(ai_vmm_core PRIVATE ws2_32)
endif()

if(UNIX)
    target_link_libraries(ai_vmm_core PRIVATE pthread dl)
endif()

# Export symbols for shared library
set_target_properties(ai_vmm_core PROPERTIES
    CXX_VISIBILITY_PRESET hidden
    VISIBILITY_INLINES_HIDDEN ON
    POSITION_INDEPENDENT_CODE ON
)