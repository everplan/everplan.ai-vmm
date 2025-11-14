add_library(ai_vmm_scheduling OBJECT
    workload_scheduler.cpp
)

target_include_directories(ai_vmm_scheduling PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/../../include
)

set_target_properties(ai_vmm_scheduling PROPERTIES
    POSITION_INDEPENDENT_CODE ON
)