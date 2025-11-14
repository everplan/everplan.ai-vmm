add_subdirectory(basic_usage)

# Add other examples when they exist
if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/multi_model_serving)
    add_subdirectory(multi_model_serving)
endif()

if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/performance_comparison)
    add_subdirectory(performance_comparison)
endif()