add_executable(ai_vmm_performance_comparison main.cpp)

target_link_libraries(ai_vmm_performance_comparison
    PRIVATE
        ai-vmm
)

target_include_directories(ai_vmm_performance_comparison
    PRIVATE
        ${CMAKE_SOURCE_DIR}/include
)
