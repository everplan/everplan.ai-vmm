find_package(GTest)

if(NOT GTest_FOUND)
    include(FetchContent)
    FetchContent_Declare(
        googletest
        URL https://github.com/google/googletest/archive/03597a01ee50f33f9bf776f968312c4036c62e5d.zip
    )
    FetchContent_MakeAvailable(googletest)
endif()

add_executable(ai_vmm_tests
    test_main.cpp
    test_tensor.cpp
    test_vmm_basic.cpp
    test_hardware_discovery.cpp
)

target_link_libraries(ai_vmm_tests PRIVATE
    ai-vmm
    gtest
    gtest_main
)

target_include_directories(ai_vmm_tests PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../include
)

include(GoogleTest)
gtest_discover_tests(ai_vmm_tests)