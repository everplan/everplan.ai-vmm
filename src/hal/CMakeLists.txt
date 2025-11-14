add_library(ai_vmm_hal OBJECT
    backend_registry.cpp
)

target_include_directories(ai_vmm_hal PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/../../include
)

set_target_properties(ai_vmm_hal PROPERTIES
    POSITION_INDEPENDENT_CODE ON
)