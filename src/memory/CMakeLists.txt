add_library(ai_vmm_memory OBJECT
    memory_manager.cpp
)

target_include_directories(ai_vmm_memory PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/../../include
)

set_target_properties(ai_vmm_memory PROPERTIES
    POSITION_INDEPENDENT_CODE ON
)