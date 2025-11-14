add_library(ai_vmm_optimization OBJECT
    model_optimizer.cpp
)

target_include_directories(ai_vmm_optimization PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/../../include
)

set_target_properties(ai_vmm_optimization PROPERTIES
    POSITION_INDEPENDENT_CODE ON
)