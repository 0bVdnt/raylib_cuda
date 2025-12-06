# tests/CMakeLists.txt

# Helper function to create test executables
function(add_rlc_test name)
    add_executable(${name} ${name}.cu)
    target_link_libraries(${name} PRIVATE raylib_cuda)
    set_target_properties(${name} PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin/tests"
    )
    if(MSVC)
        set_property(TARGET ${name} PROPERTY MSVC_RUNTIME_LIBRARY "MultiThreadedDLL")
    endif()
endfunction()

# Test executables
add_rlc_test(test_basic)
add_rlc_test(test_stress)
add_rlc_test(test_visual)