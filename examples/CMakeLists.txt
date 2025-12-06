add_executable(simple simple.cu)
target_link_libraries(simple PRIVATE raylib_cuda)

set_target_properties(simple PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)