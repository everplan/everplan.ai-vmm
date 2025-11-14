add_executable(ai_vmm_basic_example main.cpp)

target_link_libraries(ai_vmm_basic_example PRIVATE
    ai-vmm
)

target_include_directories(ai_vmm_basic_example PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../../include
)